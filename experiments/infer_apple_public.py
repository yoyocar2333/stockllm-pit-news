"""Text-only local FinBERT + spaCy features, with no market-data access.

All body tokens are scored in nonoverlapping chunks. FinBERT probabilities are
sentiment outputs; supervised next-session probabilities are fitted elsewhere.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from event_extraction.extractors import SpacyExtractor

TYPES=('litigation','earnings','product_launch','merger_acquisition','regulation','other')
ROLES=('actor','object','both','other','unknown')
DIMS={'actor':8,'action':16,'object':16}
IMPORTANCE={'earnings':2,'regulation':2,'litigation':2,'merger_acquisition':2,'product_launch':1.5,'other':1}
EVENT_COLUMNS=([f'event_type_{key}' for key in TYPES]+[f'target_role_{key}' for key in ROLES]+
               ['event_present','event_log_count']+
               [f'{field}_hash_{i}' for field,size in DIMS.items() for i in range(size)])


def event_features(events):
    events=[e for e in events if e['extraction_status']=='ok']
    n=len(events)
    vector={column:0.0 for column in EVENT_COLUMNS}
    vector['event_present']=float(n>0)
    vector['event_log_count']=float(np.log1p(n))
    for e in events:
        vector['event_type_'+e['event_type']]+=1/max(n,1)
        vector['target_role_'+e['target_role']]+=1/max(n,1)
        for field,size in DIMS.items():
            value=' '.join(str(e[field]).casefold().split())
            if not value:
                continue
            digest=hashlib.sha256((field+':'+value).encode()).digest()
            index=int.from_bytes(digest[:4],'big')%size
            sign=1 if digest[4]%2 else -1
            vector[f'{field}_hash_{index}']+=sign/max(n,1)
    target_fraction=sum(e['target_role'] in {'actor','object','both'} for e in events)/max(n,1)
    importance=sum(IMPORTANCE[e['event_type']] for e in events)/n if n else 1.0
    return vector,(.25+.75*target_fraction)*importance,n


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/raw/apple_public_v1/apple_newsroom_articles.csv')
    p.add_argument('--output-dir',default='outputs/apple_public_v1')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--revision',default='0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031')
    args=p.parse_args()
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input,dtype=str,keep_default_na=False)
    required={'article_id','title','content','publication_date'}
    if required-set(df.columns) or df.article_id.duplicated().any() or df.empty:
        raise ValueError('Invalid text input')
    input_sha=hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(args.threads)
    model_name='ProsusAI/finbert'
    config_path=Path(hf_hub_download(model_name,'config.json',revision=args.revision))
    revision=config_path.parent.name
    if not re.fullmatch(r'[0-9a-f]{40}',revision):
        raise ValueError('Cannot identify an exact cached model revision')
    tokenizer=AutoTokenizer.from_pretrained(model_name,revision=revision)
    model=AutoModelForSequenceClassification.from_pretrained(model_name,revision=revision)
    model.eval()
    labels={str(label).lower():int(index) for index,label in model.config.id2label.items()}
    if set(labels)!={'positive','negative','neutral'}:
        raise ValueError('Unexpected FinBERT labels')
    registry={'input_sha256':input_sha,'model':model_name,'revision':revision,
              'tokenizer':type(tokenizer).__name__,'pooling':'token-weighted nonoverlapping chunks',
              'packages':{name:version(name) for name in ('torch','transformers','spacy','en-core-web-sm')},
              'text_input_only':True,'market_input_to_extraction':False}
    registry_path=output/'model_registry.json'
    if registry_path.exists():
        old=json.loads(registry_path.read_text(encoding='utf-8'))
        if old['input_sha256']!=input_sha or old['revision']!=revision:
            raise ValueError('Cache population/model differs; use a new output directory')
    registry_path.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    cache_path=output/'finbert_chunks.jsonl'
    cache={}
    if cache_path.exists():
        for line in cache_path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line); cache[(row['article_id'],row['source'],row['chunk'])]=row
    tasks=[]; lengths={}; chunks_per={}
    for row in df.to_dict('records'):
        for source,text in [('title',row['title']),('full',row['title']+'\n\n'+row['content'])]:
            ids=tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)['input_ids']
            if not ids:
                raise ValueError('Empty text')
            chunks=[ids[start:start+510] for start in range(0,len(ids),510)]
            lengths[(row['article_id'],source)]=len(ids)
            chunks_per[(row['article_id'],source)]=len(chunks)
            for index,chunk in enumerate(chunks):
                key=(row['article_id'],source,index)
                if key not in cache:
                    tasks.append((key,[tokenizer.cls_token_id,*chunk,tokenizer.sep_token_id],len(chunk)))
    # Similar-length batches avoid padding headlines to the full body length.
    tasks.sort(key=lambda task:len(task[1]))
    start_time=time.monotonic()
    print('Articles:',len(df),'uncached chunks:',len(tasks),'threads:',args.threads,flush=True)
    with torch.inference_mode(),cache_path.open('a',encoding='utf-8') as handle:
        for begin in range(0,len(tasks),args.batch_size):
            batch=tasks[begin:begin+args.batch_size]
            longest=max(len(task[1]) for task in batch)
            if longest>model.config.max_position_embeddings:
                raise ValueError('Chunk exceeds the model position limit')
            input_ids=torch.full((len(batch),longest),tokenizer.pad_token_id,dtype=torch.long)
            attention=torch.zeros_like(input_ids)
            for index,task in enumerate(batch):
                input_ids[index,:len(task[1])]=torch.tensor(task[1],dtype=torch.long)
                attention[index,:len(task[1])]=1
            inputs={'input_ids':input_ids,'attention_mask':attention}
            probabilities=torch.softmax(model(**inputs).logits,dim=-1).numpy()
            for (key,ids,length),probs in zip(batch,probabilities):
                record=dict(article_id=key[0],source=key[1],chunk=key[2],tokens=length,
                            positive=float(probs[labels['positive']]),negative=float(probs[labels['negative']]),
                            neutral=float(probs[labels['neutral']]))
                cache[key]=record
                handle.write(json.dumps(record)+'\n')
            handle.flush()
            if begin==0 or (begin//args.batch_size+1)%10==0 or begin+len(batch)==len(tasks):
                done=begin+len(batch)
                print('FinBERT chunks:',done,'/',len(tasks),'elapsed seconds:',round(time.monotonic()-start_time,1),flush=True)
    extractor=SpacyExtractor()
    rows=[]; status_counts={source:Counter() for source in ('title','full')}
    for i,row in enumerate(df.to_dict('records'),1):
        for source in ('title','full'):
            key=(row['article_id'],source)
            records=[cache[(key[0],key[1],j)] for j in range(chunks_per[key])]
            total=sum(record['tokens'] for record in records)
            probabilities={name:sum(record[name]*record['tokens'] for record in records)/total
                           for name in ('positive','negative','neutral')}
            texts=[row['title']] if source=='title' else [row['title'],row['content']]
            events=[event for text in texts for event in extractor.extract(text,'Apple')]
            status_counts[source].update(e['extraction_status'] for e in events)
            features,weight,n_events=event_features(events)
            rows.append(dict(article_id=row['article_id'],publication_date=row['publication_date'],source=source,
                             finbert_tokens=lengths[key],finbert_chunks=chunks_per[key],truncated_tokens=0,
                             chunking_needed=int(chunks_per[key]>1),event_records=n_events,article_weight=weight,
                             **{'sent_'+name:value for name,value in probabilities.items()},**features))
        if i%50==0:
            print('spaCy articles:',i,'/',len(df),flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(output/'article_features.csv',index=False)
    audit={'articles':len(df),'feature_rows':len(result),'input_sha256':input_sha,
           'full_articles_over_single_512_token_window':int(result[result.source=='full'].chunking_needed.sum()),
           'total_body_and_title_chunks':int(result.finbert_chunks.sum()),'truncated_tokens':0,
           'extraction_status_counts':{source:dict(count) for source,count in status_counts.items()},
           'title_articles_without_event':int((result[result.source=='title'].event_records==0).sum()),
           'full_articles_without_event':int((result[result.source=='full'].event_records==0).sum()),
           'independent_human_gold_available':False}
    (output/'inference_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    main()
