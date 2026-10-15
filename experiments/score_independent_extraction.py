"""Score the frozen blind title sample only after independent labels are supplied.

Reviewer independence is declared by the annotator, not authenticated by this
script. Pending or AI-assisted labels cannot be reported as human validation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from event_extraction.extractors import Event

FIELDS=('actor','action','object','event_type','target_role')
TYPES={'litigation','earnings','product_launch','merger_acquisition','regulation','other'}
ROLES={'actor','object','both','other','unknown'}


def load_reference(path,expected):
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(rows)!=len(expected) or {row.get('article_id') for row in rows}!=set(expected):
        raise ValueError('Gold IDs must exactly match the frozen sample without duplicates')
    for row in rows:
        if row.get('review_status')!='reviewed':
            raise ValueError('Independent labels are still pending')
        if row.get('reviewer_type')!='human_independent' or not str(row.get('reviewer_id','')).strip():
            raise ValueError('Independent human reviewer declaration is missing')
        if row.get('ai_exposed'):
            raise ValueError('AI-exposed labels cannot be scored as independent human labels')
        if row.get('source')!='title' or not isinstance(row.get('no_event'),bool):
            raise ValueError('Only title labels with explicit boolean no_event are supported')
        events=row.get('events')
        if not isinstance(events,list) or row['no_event']!= (len(events)==0):
            raise ValueError('no_event and events disagree')
        for event in events:
            if any(field not in event or not isinstance(event[field],str) for field in FIELDS):
                raise ValueError('Every gold event needs five string fields')
            if not event['action'].strip() or not (event['actor'].strip() or event['object'].strip()):
                raise ValueError('Gold event needs action and at least one argument')
            if event['event_type'] not in TYPES or event['target_role'] not in ROLES:
                raise ValueError('Invalid gold type/role')
    return {row['article_id']:row for row in rows}


def event_key(event):
    return tuple(' '.join(str(event.get(field,'')).casefold().split()) for field in FIELDS)


def score_events(predictions,reference):
    predicted=Counter(event_key(e) for e in predictions)
    gold=Counter(event_key(e) for e in reference)
    tp=sum((predicted&gold).values()); n_pred=sum(predicted.values()); n_gold=sum(gold.values())
    return {'tp':tp,'predicted':n_pred,'gold':n_gold}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold',required=True)
    p.add_argument('--audit-dir',default='outputs/apple_extraction_audit_v1')
    args=p.parse_args()
    folder=Path(args.audit_dir)
    frozen=json.loads((folder/'sample_freeze.json').read_text())['selected_ids']
    gold=load_reference(args.gold,frozen)
    v1={row['article_id']:row['title'] for row in map(json.loads,(folder/'private_development_events.jsonl').read_text(encoding='utf-8').splitlines())}
    v2={row['article_id']:row['kept'] for row in json.loads((folder/'private_candidate_events.json').read_text(encoding='utf-8')) if row['source']=='title'}
    backends={'v1':v1,'v2_candidate':v2}
    gliner_path=folder/'private_gliner_title_events.json'
    if gliner_path.exists():
        backends['gliner2']={row['article_id']:row['events'] for row in json.loads(gliner_path.read_text(encoding='utf-8'))}
    results={}
    for name,events_by_id in backends.items():
        totals=Counter(tp=0,predicted=0,gold=0)
        for article_id in frozen:
            if article_id not in events_by_id:
                raise ValueError('Missing backend article: '+article_id)
            totals.update(score_events(events_by_id[article_id],gold[article_id]['events']))
        tp,n_pred,n_gold=totals['tp'],totals['predicted'],totals['gold']
        results[name]={**totals,'precision':tp/n_pred if n_pred else None,'recall':tp/n_gold if n_gold else None,
                       'f1':2*tp/(n_pred+n_gold) if n_pred+n_gold else None}
    report={'articles':len(frozen),'reviewer_independence':'declared by annotators, not authenticated',
            'metric':'strict normalized five-field multiset agreement, not span-overlap F1','backends':results}
    (folder/'independent_scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
