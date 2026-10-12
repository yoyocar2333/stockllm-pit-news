"""Export spreadsheet-friendly human labels and score article-level event sets.

Use --blind for an independent reference set. Assisted review is for debugging.
Neither path labels financial impact or future prices.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pandas as pd

FIELDS = ('actor', 'action', 'object', 'event_type', 'target_role')
TYPES = {'litigation', 'earnings', 'product_launch', 'merger_acquisition', 'regulation', 'other'}
ROLES = {'actor', 'object', 'both', 'other', 'unknown'}
BASE = ('article_id', 'ticker', 'target_name', 'title', 'content', 'source')


def load_predictions(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = set(BASE) | {'events_json'}
    if required - set(df.columns):
        raise ValueError('Missing prediction columns: ' + ', '.join(sorted(required - set(df.columns))))
    if df.empty or df.article_id.str.strip().eq('').any() or df.article_id.duplicated().any():
        raise ValueError('Predictions require nonempty, unique article_id values')
    if not set(df.source) <= {'title', 'full'}:
        raise ValueError('Invalid prediction source')
    articles = {}
    for row in df.to_dict('records'):
        events = json.loads(row['events_json'])
        if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
            raise ValueError('events_json must be a list of objects')
        row['events'] = [e for e in events if e.get('extraction_status') == 'ok']
        articles[row['article_id']] = row
    return articles


def prepare(input_path, output_path, blind=False):
    articles = load_predictions(input_path)
    rows = []
    for article in articles.values():
        events = [] if blind else article['events']
        for index, event in enumerate(events or [{}]):
            row = {key: article[key] for key in BASE}
            # Full-body text is hidden in title-only annotation to preserve the arm.
            if article['source'] == 'title':
                row['content'] = ''
            row.update(row_id=f"{article['article_id']}:{index}",
                       annotation_mode='blind' if blind else 'assisted',
                       review_status='pending', decision='', evidence='', notes='')
            row.update({f'predicted_{key}': event.get(key, '') for key in FIELDS})
            row.update({f'gold_{key}': '' for key in FIELDS})
            rows.append(row)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Refusing to overwrite existing annotation: ' + str(output))
    pd.DataFrame(rows).to_csv(output, index=False, encoding='utf-8-sig')
    print('Saved:', output, 'rows:', len(rows), 'mode:', 'blind' if blind else 'assisted')


def normalized(value):
    return ' '.join(str(value).casefold().split())


def key(event, fields):
    return tuple(normalized(event.get(field, '')) for field in fields)


def score(predictions_path, annotations_path):
    articles = load_predictions(predictions_path)
    df = pd.read_csv(annotations_path, dtype=str, keep_default_na=False)
    required = {'article_id', 'source', 'row_id', 'annotation_mode', 'review_status',
                'decision', 'evidence'} | {f'gold_{k}' for k in FIELDS}
    if required - set(df.columns):
        raise ValueError('Missing annotation columns: ' + ', '.join(sorted(required - set(df.columns))))
    if df.empty or df.row_id.str.strip().eq('').any() or df.row_id.duplicated().any():
        raise ValueError('Each annotation row needs a unique nonempty row_id')
    if set(df.article_id) != set(articles):
        raise ValueError('Annotation article IDs must exactly match prediction IDs; do not drop articles')
    modes = set(df.annotation_mode)
    if len(modes) != 1 or not modes <= {'blind', 'assisted'}:
        raise ValueError('Use one annotation mode per file: blind or assisted')
    if not set(df.review_status) <= {'reviewed', 'uncertain'}:
        raise ValueError('Finish all pending rows first: review_status=reviewed or uncertain')
    totals = {name: Counter(tp=0, predicted=0, gold=0) for name in ('triple', 'all_fields')}
    excluded = []
    scored = 0
    for article_id, group in df.groupby('article_id', sort=False):
        article = articles[article_id]
        if set(group.source) != {article['source']}:
            raise ValueError('Annotation source differs for ' + article_id)
        if 'uncertain' in set(group.review_status):
            excluded.append(article_id)
            continue
        if not set(group.decision) <= {'keep', 'delete', 'add', 'no_event'}:
            raise ValueError('decision must be keep/delete/add/no_event: ' + article_id)
        gold = []
        for row in group.to_dict('records'):
            if row['decision'] in {'delete', 'no_event'}:
                if any(row[f'gold_{field}'].strip() for field in FIELDS):
                    raise ValueError('Clear gold fields on delete/no_event rows: ' + row['row_id'])
                continue
            event = {field: row[f'gold_{field}'].strip() for field in FIELDS}
            if not event['action'] or not (event['actor'] or event['object']):
                raise ValueError('Gold event needs action and at least actor/object: ' + row['row_id'])
            if event['event_type'] not in TYPES or event['target_role'] not in ROLES:
                raise ValueError('Invalid gold category/role: ' + row['row_id'])
            evidence = row['evidence'].strip()
            text = article['title'] if article['source'] == 'title' else article['title'] + '\n\n' + article['content']
            if not evidence or evidence not in text:
                raise ValueError('Evidence must be an exact excerpt of input text: ' + row['row_id'])
            gold.append(event)
        if gold and 'no_event' in set(group.decision):
            raise ValueError('no_event contradicts gold events: ' + article_id)
        scored += 1
        for name, fields in [('triple', FIELDS[:3]), ('all_fields', FIELDS)]:
            predicted = Counter(key(e, fields) for e in article['events'])
            reference = Counter(key(e, fields) for e in gold)
            totals[name].update(tp=sum((predicted & reference).values()),
                                predicted=sum(predicted.values()), gold=sum(reference.values()))
    if not scored:
        raise ValueError('No reviewed articles remain after excluding uncertain articles')
    result = {'annotation_mode': next(iter(modes)), 'scored_articles': scored,
              'excluded_uncertain_articles': excluded,
              'note': 'Exact normalized set matching; assisted review is not independent validation.'}
    for name, counts in totals.items():
        tp, pred, gold = counts['tp'], counts['predicted'], counts['gold']
        result[name] = {**counts, 'false_positive': pred-tp, 'false_negative': gold-tp,
                        'precision': tp/pred if pred else None,
                        'recall': tp/gold if gold else None,
                        'f1': 2*tp/(pred+gold) if pred+gold else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    export = subs.add_parser('prepare')
    export.add_argument('--input', required=True)
    export.add_argument('--output', required=True)
    export.add_argument('--blind', action='store_true')
    check = subs.add_parser('validate')
    check.add_argument('--predictions', required=True)
    check.add_argument('--annotations', required=True)
    check.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.input, args.output, args.blind)
    else:
        result = score(args.predictions, args.annotations)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
