"""Compare V1/V2 structural diagnostics; optionally GLiNER2 on frozen titles.

Counts/disagreements are not an accuracy benchmark without independent labels.
All processing is restricted to the frozen 2023 sample. No price labels are read.
"""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import hashlib
import pandas as pd
from event_extraction.candidate_v2 import filter_candidates


def key(event):
    return tuple(' '.join(str(event[field]).casefold().split()) for field in ('actor','action','object','event_type','target_role'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',default='outputs/apple_extraction_audit_v1')
    parser.add_argument('--input',default='data/raw/apple_public_v1/apple_newsroom_articles.csv')
    parser.add_argument('--compare-gliner',action='store_true')
    args=parser.parse_args()
    output=Path(args.audit_dir)
    freeze=json.loads((output/'sample_freeze.json').read_text())
    if hashlib.sha256(Path(args.input).read_bytes()).hexdigest()!=freeze['source_input_sha256']:
        raise ValueError('Source changed since sample freeze')
    selected=set(freeze['selected_ids'])
    source=pd.read_csv(args.input,dtype=str,keep_default_na=False).set_index('article_id')
    data=[json.loads(line) for line in (output/'private_development_events.jsonl').read_text(encoding='utf-8').splitlines()]
    if any(not source.loc[row['article_id'],'publication_date'].startswith('2023-') for row in data):
        raise ValueError('Only 2023 allowed')
    rows=[]; counts={name:Counter() for name in ('title','full')}; private=[]
    for row in data:
        for name in ('title','full'):
            kept,rejected=filter_candidates(row[name])
            counts[name].update(v1_records=len(row[name]),v2_candidates=len(kept),
                                v2_partial=sum(e['extraction_status']=='partial_requires_review' for e in kept),
                                v2_negated=sum(e['negated'] for e in kept),
                                v2_modal=sum(bool(e['modals']) for e in kept),
                                company_role_changed=sum(e['target_role']!=e['v1_target_role'] for e in kept),
                                event_type_changed=sum(e['event_type']!=e['v1_event_type'] for e in kept))
            for e in rejected:
                counts[name].update('rejected_'+reason for reason in e['rejection_reasons'])
            rows.append({'article_id':row['article_id'],'source':name,'url':row['url'],
                         'v1_records':len(row[name]),'v2_candidates':len(kept),
                         'v2_partial':sum(e['extraction_status']=='partial_requires_review' for e in kept),
                         'in_frozen_sample':row['article_id'] in selected})
            private.append({'article_id':row['article_id'],'source':name,'kept':kept,'rejected':rejected})
    (output/'private_candidate_events.json').write_text(json.dumps(private,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame(rows).to_csv(output/'candidate_comparison.csv',index=False)
    summary={'scope_year':2023,'development_articles':len(data),'frozen_sample_size':len(selected),
             'structural_counts':{name:dict(value) for name,value in counts.items()},
             'not_accuracy':'Filters can remove valid events; independent precision/recall is unknown',
             'packages':{name:version(name) for name in ('spacy','en-core-web-sm')}}
    if args.compare_gliner:
        import torch
        from huggingface_hub import hf_hub_download
        from event_extraction.extractors import GLiNER2Extractor
        torch.set_num_threads(4)
        config=Path(hf_hub_download('fastino/gliner2-base-v1','config.json'))
        extractor=GLiNER2Extractor()
        comparisons=[]; private_gliner=[]
        for i,row in enumerate([row for row in data if row['article_id'] in selected],1):
            text=source.loc[row['article_id'],'title']
            events=[e for e in extractor.extract(text,'Apple') if e['extraction_status']=='ok']
            a,b=set(map(key,row['title'])),set(map(key,events))
            comparisons.append({'article_id':row['article_id'],'url':row['url'],
                                'v1_title_records':len(row['title']),'gliner_title_records':len(events),
                                'v1_incomplete_triples':sum(not e['actor'] or not e['object'] for e in row['title']),
                                'gliner_incomplete_triples':sum(not e['actor'] or not e['object'] for e in events),
                                'gliner_missing_action':sum(not e['action'].strip() for e in events),
                                'exact_shared_event_keys':len(a&b),'outputs_disagree':a!=b})
            private_gliner.append({'article_id':row['article_id'],'events':events})
            if i%20==0: print('GLiNER2 title checks:',i,'/',len(selected),flush=True)
        comparison=pd.DataFrame(comparisons)
        comparison.to_csv(output/'gliner_disagreements.csv',index=False)
        (output/'private_gliner_title_events.json').write_text(json.dumps(private_gliner,ensure_ascii=False),encoding='utf-8')
        summary['gliner']={'model':'fastino/gliner2-base-v1','revision':config.parent.name,
                           'package_version':version('gliner2'),'source':'title_only',
                           'articles':len(comparison),'article_outputs_disagree':int(comparison.outputs_disagree.sum()),
                           'v1_no_event_articles':int((comparison.v1_title_records==0).sum()),
                           'gliner_no_event_articles':int((comparison.gliner_title_records==0).sum()),
                           'gliner_missing_action_records':int(comparison.gliner_missing_action.sum()),
                           'gliner_incomplete_triple_records':int(comparison.gliner_incomplete_triples.sum()),
                           'v1_incomplete_triple_records':int(comparison.v1_incomplete_triples.sum()),
                           'note':'Disagreement is not proof either backend is correct; class rules are shared.'}
    (output/'candidate_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
