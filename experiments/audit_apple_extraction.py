"""Price-free structural diagnostics on 2023 development text only.

Flags are review signals, not ground-truth errors. Full evidence stays local;
public outputs contain counts, URLs and short identifiers, not article bodies.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import pandas as pd
from event_extraction.extractors import Event, SpacyExtractor, event_type, target_role

PRONOUNS={'we','i','you','he','she','it','they','who','this','that','these','those'}
REPORTING={'say','tell','add','note','thank','explain','believe','hope','think'}
MODALS={'may','might','could','would','should','must','can','will'}


def phrase(token):
    return ' '.join(t.text for t in token.subtree) if token is not None else ''


def diagnose_segment(nlp,text,target_name='Apple'):
    if not text.strip():
        return []
    doc=nlp(text)
    chunks={token.i:chunk.text for chunk in doc.noun_chunks for token in [chunk.root]}
    events=[]
    for sentence in doc.sents:
        for verb in sentence:
            if verb.pos_ not in {'VERB','AUX'}:
                continue
            children=list(verb.children)
            subjects=[c for c in children if c.dep_ in ('nsubj','csubj','nsubjpass')]
            objects=[c for c in children if c.dep_ in ('dobj','obj','dative','attr')]
            agents=[z for c in children if c.dep_=='agent' for z in c.children if z.dep_=='pobj']
            subject=subjects[0] if subjects else None
            obj=objects[0] if objects else None
            passive=any(s.dep_=='nsubjpass' for s in subjects)
            actor_token,object_token=(agents[0] if agents else None,subject) if passive else (subject,obj)
            actor,affected=phrase(actor_token),phrase(object_token)
            if not actor and not affected:
                continue
            record=asdict(Event(actor=actor,action=verb.lemma_,object=affected,
                                 event_type=event_type(verb.lemma_,sentence.text),
                                 target_role=target_role(actor,affected,target_name)))
            # Extra diagnostics only; do not change any reproduced V1 field.
            conj_actor=''
            if not actor and verb.dep_=='conj' and verb.head.pos_=='VERB' and not passive:
                inherited=[c for c in verb.head.children if c.dep_ in ('nsubj','csubj')]
                if inherited: conj_actor=phrase(inherited[0])
            flags=[]
            if verb.pos_=='AUX': flags.append('auxiliary_as_event')
            if verb.lemma_.casefold() in REPORTING: flags.append('reporting_or_mental_verb')
            if not actor or not affected: flags.append('incomplete_triple')
            if max(len(actor.split()),len(affected.split()))>20: flags.append('long_argument_span')
            if actor_token is not None and actor_token.text.casefold() in PRONOUNS: flags.append('pronoun_actor')
            if record['target_role'] in {'other','unknown'}: flags.append('no_explicit_target_argument')
            core_actor=chunks.get(actor_token.i,actor_token.text) if actor_token is not None else ''
            core_object=chunks.get(object_token.i,object_token.text) if object_token is not None else ''
            core_role=target_role(core_actor,core_object,target_name)
            if record['target_role'] in {'actor','object','both'} and core_role in {'other','unknown'}:
                flags.append('target_only_in_argument_subtree')
            negated=any(c.dep_=='neg' for c in children)
            modals=[c.lemma_.casefold() for c in children if c.dep_ in {'aux','auxpass'} and c.lemma_.casefold() in MODALS]
            if negated: flags.append('negation_not_in_v1_schema')
            if modals: flags.append('modality_not_in_v1_schema')
            events.append({**record,'flags':flags,'verb_pos':verb.pos_,
                           'negated':negated,'modals':modals,'sentence':sentence.text,
                           'verb_start':verb.idx,'verb_end':verb.idx+len(verb.text),
                           'core_actor':core_actor,'core_object':core_object})
            events[-1]['conj_actor']=conj_actor
    return events


def summarize(records):
    counts=Counter(flag for event in records for flag in event['flags'])
    actions=Counter(e['action'] for e in records)
    types=Counter(e['event_type'] for e in records)
    return {'events':len(records),'flag_counts':dict(counts),
            'event_type_counts':dict(types),'top_actions':actions.most_common(12)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/raw/apple_public_v1/apple_newsroom_articles.csv')
    p.add_argument('--feature-reference',default='outputs/apple_public_v1/article_features.csv')
    p.add_argument('--output-dir',default='outputs/apple_extraction_audit_v1')
    p.add_argument('--sample-size',type=int,default=100)
    p.add_argument('--seed',type=int,default=20261014)
    args=p.parse_args()
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(args.input,dtype=str,keep_default_na=False)
    # Filter before loading any text into the model. No 2024/2025 diagnostics.
    dev=raw[raw.publication_date.str.startswith('2023-')].copy().sort_values('article_id').reset_index(drop=True)
    if len(dev)<args.sample_size or dev.article_id.duplicated().any():
        raise ValueError('Insufficient or ambiguous development population')
    sample=dev.sample(args.sample_size,random_state=args.seed).sort_values('article_id')
    existing_freeze=output/'sample_freeze.json'
    input_hash=hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    if existing_freeze.exists():
        old=json.loads(existing_freeze.read_text())
        if old['selected_ids']!=sample.article_id.tolist() or old['source_input_sha256']!=input_hash:
            raise ValueError('Frozen sample or source differs; choose a new output directory')
    columns=['article_id','publication_date','url','text_sha256']
    sample[columns].to_csv(output/'blind_review_manifest.csv',index=False)
    freeze={'seed':args.seed,'sample_size':len(sample),'population_year':2023,
            'population_size':len(dev),'selected_ids':sample.article_id.tolist(),
            'source_input_sha256':input_hash,
            'reviewer':'not yet independently human annotated','selection':'uniform random without replacement, before diagnostics'}
    (output/'sample_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    model=SpacyExtractor()
    private_path=output/'private_development_events.jsonl'
    all_records={'title':[],'full':[]}; sampled_records={'title':[],'full':[]}
    article_stats=[]
    reference=pd.read_csv(args.feature_reference).set_index(['article_id','source'])
    with private_path.open('w',encoding='utf-8') as handle:
        for index,row in enumerate(dev.to_dict('records'),1):
            title=diagnose_segment(model.nlp,row['title'])
            full=title+diagnose_segment(model.nlp,row['content'])
            handle.write(json.dumps({'article_id':row['article_id'],'url':row['url'],'title':title,'full':full},ensure_ascii=False)+'\n')
            for source,events in [('title',title),('full',full)]:
                expected=int(reference.loc[(row['article_id'],source),'event_records'])
                if len(events)!=expected:
                    raise ValueError('V1 reproduction mismatch: '+row['article_id']+'/'+source)
                all_records[source].extend(events)
                if row['article_id'] in set(sample.article_id): sampled_records[source].extend(events)
                unique={tuple(' '.join(str(e[k]).casefold().split()) for k in ('actor','action','object','event_type')) for e in events}
                article_stats.append({'article_id':row['article_id'],'publication_date':row['publication_date'],
                                      'url':row['url'],'source':source,'events':len(events),
                                      'exact_duplicate_mentions':len(events)-len(unique),
                                      'events_with_any_flag':sum(bool(e['flags']) for e in events)})
            if index%25==0: print('Development articles:',index,'/',len(dev),flush=True)
    stats=pd.DataFrame(article_stats)
    stats.to_csv(output/'development_diagnostics.csv',index=False)
    # Blank labels are kept in JSONL for multi-event, independent human review.
    # Reviewers follow the URLs; no model outputs or article bodies are embedded.
    gold_path=output/'independent_gold_template.jsonl'
    if not gold_path.exists():
        with gold_path.open('w',encoding='utf-8') as handle:
            for row in sample.to_dict('records'):
                handle.write(json.dumps({'article_id':row['article_id'],'url':row['url'],
                                        'ticker':'AAPL','target_name':'Apple',
                                        'source':'title','review_status':'pending','reviewer_id':'',
                                        'reviewer_type':'','no_event':None,
                                        'events':[],'notes':''})+'\n')
    else:
        print('Existing independent gold template preserved:',gold_path,flush=True)
    summary={'scope':'2023 development only; price-free; no 2024/2025 text used for diagnostics',
             'development_articles':len(dev),'frozen_blind_sample_articles':len(sample),
             'v1_event_counts_reproduced':True,'independent_human_gold_available':False,
             'all_development':{source:summarize(records) for source,records in all_records.items()},
             'frozen_sample':{source:summarize(records) for source,records in sampled_records.items()},
             'exact_duplicate_mentions':stats.groupby('source').exact_duplicate_mentions.sum().to_dict(),
             'interpretation':'Flags are structural review indicators, not extraction error rates or precision/recall.'}
    (output/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
