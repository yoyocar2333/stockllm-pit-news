"""Price-free AI-assisted title review and a separate, blank human review page.

No independent-human scores or forecasts are produced. Source titles, semantic
drafts and the generated review page stay under ignored outputs/.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from experiments.score_independent_extraction import FIELDS, ROLES, TYPES, event_key

POLICY = 'title_semantic_v1_20261015'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_snapshot(input_path, audit_dir):
    freeze = json.loads((Path(audit_dir) / 'sample_freeze.json').read_text(encoding='utf-8'))
    input_path = Path(input_path)
    if sha256(input_path.read_bytes()) != freeze['source_input_sha256']:
        raise ValueError('Source differs from frozen sample')
    with input_path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len({r['article_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source article IDs')
    by_id = {r['article_id']: r for r in rows}
    ids = freeze['selected_ids']
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError('This gate requires the frozen 100-item sample')
    selected = [by_id[article_id] for article_id in ids]
    if any(not r['publication_date'].startswith('2023-') for r in selected):
        raise ValueError('Only 2023 development titles are allowed')
    return freeze, selected


def validate_events(row):
    events = row.get('events')
    if not isinstance(events, list) or not isinstance(row.get('no_event'), bool):
        raise ValueError('Explicit no_event and event list required')
    if row['no_event'] != (len(events) == 0):
        raise ValueError('no_event and events disagree')
    for event in events:
        if not isinstance(event, dict) or any(not isinstance(event.get(f), str) for f in FIELDS):
            raise ValueError('Five string event fields required')
        if not event['action'].strip() or not (event['actor'].strip() or event['object'].strip()):
            raise ValueError('Action and at least one argument required')
        if event['event_type'] not in TYPES or event['target_role'] not in ROLES:
            raise ValueError('Invalid event class or company role')


def load_ai_draft(path, freeze, articles):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    expected = {r['article_id']: r for r in articles}
    if len(rows) != len(expected) or {r.get('article_id') for r in rows} != set(expected):
        raise ValueError('Draft IDs must match frozen sample exactly without duplicates')
    for row in rows:
        article = expected[row['article_id']]
        if row.get('reviewer_type') != 'ai_assisted' or row.get('review_status') != 'reviewed':
            raise ValueError('Only explicitly AI-assisted reviewed drafts accepted here')
        if not str(row.get('reviewer_id', '')).strip() or row.get('source') != 'title':
            raise ValueError('Reviewer and title source required')
        if row.get('annotation_policy') != POLICY:
            raise ValueError('Unexpected semantic review policy')
        if row.get('url') != article['url'] or row.get('title_sha256') != sha256(article['title'].encode('utf-8')):
            raise ValueError('Draft does not match frozen title snapshot')
        if row.get('source_input_sha256') != freeze['source_input_sha256']:
            raise ValueError('Draft source hash mismatch')
        validate_events(row)
    return {r['article_id']: r for r in rows}


def compare_backend(predictions, drafts, articles):
    totals = Counter()
    details = []
    for article in articles:
        article_id = article['article_id']
        if article_id not in predictions:
            raise ValueError('Missing backend article: ' + article_id)
        pred = predictions[article_id]
        ref = drafts[article_id]['events']
        pk, rk = Counter(map(event_key, pred)), Counter(map(event_key, ref))
        pa = Counter(' '.join(e.get('action', '').casefold().split()) for e in pred if e.get('action', '').strip())
        ra = Counter(' '.join(e['action'].casefold().split()) for e in ref)
        row = dict(article_id=article_id, url=article['url'], predicted_records=len(pred),
                   ai_draft_records=len(ref), strict_shared_records=sum((pk & rk).values()),
                   shared_action_occurrences=sum((pa & ra).values()),
                   ai_action_occurrences_not_returned=sum((ra - pa).values()),
                   zero_output_with_nonempty_ai_draft=bool(ref) and not pred,
                   exact_article_agreement=pk == rk,
                   incomplete_predictions=sum(not e.get('actor') or not e.get('object') for e in pred),
                   missing_action_predictions=sum(not e.get('action', '').strip() for e in pred))
        details.append(row)
        totals.update({k: int(v) for k, v in row.items() if isinstance(v, (int, bool))})
    return dict(totals), details


def rejection_risks(candidates, drafts, articles):
    risks = []
    for article in articles:
        article_id = article['article_id']
        item = candidates[article_id]
        kept = {e['action'].casefold() for e in item['kept']}
        wanted = {e['action'].casefold() for e in drafts[article_id]['events']}
        for rejected in item['rejected']:
            action = rejected['action'].casefold()
            if action in wanted - kept:
                risks.append(dict(article_id=article_id, url=article['url'], action=action,
                                  reasons=rejected['rejection_reasons']))
    return risks


def write_review_page(articles, freeze, output, ai_drafts=None):
    payload = dict(source_input_sha256=freeze['source_input_sha256'], policy=POLICY,
                   items=[dict(article_id=a['article_id'], title=a['title'], url=a['url'],
                               publication_date=a['publication_date'],
                               title_sha256=sha256(a['title'].encode('utf-8'))) for a in articles])
    if ai_drafts is not None:
        payload['ai_draft'] = [ai_drafts[a['article_id']] for a in articles]
    template = (Path(__file__).parent.parent / 'event_extraction/review_ui.html').read_text(encoding='utf-8')
    encoded = json.dumps(payload, ensure_ascii=True).replace('<', '\\u003c')
    Path(output).write_text(template.replace('/*__REVIEW_PAYLOAD__*/ null', encoded), encoding='utf-8')


def write_blind_page(articles, freeze, output):
    # Deliberately contains no AI/model predictions. Draft import is an explicit
    # user action and permanently marks imported labels as AI-exposed.
    write_review_page(articles, freeze, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/raw/apple_public_v1/apple_newsroom_articles.csv')
    parser.add_argument('--audit-dir', default='outputs/apple_extraction_audit_v1')
    parser.add_argument('--output-dir', default='outputs/apple_extraction_review_v1')
    parser.add_argument('--ai-draft', required=True)
    args = parser.parse_args()
    audit, output = Path(args.audit_dir), Path(args.output_dir)
    freeze, articles = load_snapshot(args.input, audit)
    drafts = load_ai_draft(args.ai_draft, freeze, articles)
    v1 = {r['article_id']: r['title'] for r in map(json.loads, (audit / 'private_development_events.jsonl').read_text(encoding='utf-8').splitlines())}
    v2 = {r['article_id']: r for r in json.loads((audit / 'private_candidate_events.json').read_text(encoding='utf-8')) if r['source'] == 'title'}
    backends = {'v1': v1, 'v2_candidate': {k: v['kept'] for k, v in v2.items()}}
    gliner = audit / 'private_gliner_title_events.json'
    if gliner.exists():
        backends['gliner2'] = {r['article_id']: r['events'] for r in json.loads(gliner.read_text(encoding='utf-8'))}
    results, details = {}, {}
    for name, predictions in backends.items():
        results[name], details[name] = compare_backend(predictions, drafts, articles)
    risks = rejection_risks(v2, drafts, articles)
    summary = dict(date='2026-10-15', scope='100 frozen 2023 titles; no body or price labels used',
                   annotation_policy=POLICY, source_input_sha256=freeze['source_input_sha256'],
                   ai_reviewed_articles=len(drafts), ai_draft_event_records=sum(len(r['events']) for r in drafts.values()),
                   ai_no_event_articles=sum(r['no_event'] for r in drafts.values()),
                   ai_draft_sha256=sha256(Path(args.ai_draft).read_bytes()),
                   ai_event_class_counts=dict(Counter(e['event_type'] for r in drafts.values() for e in r['events'])),
                   independent_human_reviewed_articles=0, independent_extraction_scores_created=False,
                   eligible_for_v2_forecast_validation=False, gate_status='pending_independent_human_review',
                   forecast_retrained=False, test_2025_reopened=False, backends=results,
                   candidate_rejection_action_risk_records=len(risks),
                   candidate_rejection_action_risk_articles=len({r['article_id'] for r in risks}),
                   interpretation='Counts measure agreement with an AI-authored draft, NOT extraction accuracy. '
                                  'Action overlap alone does not validate roles, spans, factuality or event classes. '
                                  'Semantic policy includes availability/ellipsis and entity affiliation; V1 was syntactic. '
                                  'No model winner or human precision/recall/F1 is established.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'ai_review_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'private_ai_agreement_details.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    (output / 'private_rejection_action_risks.json').write_text(json.dumps(risks, indent=2), encoding='utf-8')
    write_blind_page(articles, freeze, output / 'review_titles.html')
    write_review_page(articles, freeze, output / 'review_titles_ai_assisted.html', drafts)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
