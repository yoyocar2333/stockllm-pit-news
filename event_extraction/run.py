import argparse, json, hashlib, platform
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import pandas as pd
from .extractors import SpacyExtractor, GLiNER2Extractor

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='event_extraction/sample_news.csv')
    p.add_argument('--output', default='outputs/events.csv')
    p.add_argument('--backend', choices=['spacy', 'gliner2'], default='spacy')
    p.add_argument('--source', choices=['title','full'], default='title')
    a = p.parse_args()
    # Preserve identifiers such as 00001; never infer them as numbers.
    df = pd.read_csv(a.input, dtype=str, keep_default_na=False)
    required = {'article_id','ticker','target_name','title','content'}
    if required - set(df.columns):
        raise ValueError('Missing: '+', '.join(sorted(required-set(df.columns))))
    if df.article_id.duplicated().any():
        raise ValueError('article_id must be unique')
    if df.empty:
        raise ValueError('Input contains no articles')
    for column in ('article_id', 'ticker', 'target_name'):
        if df[column].str.strip().eq('').any():
            raise ValueError(column + ' must not be empty')
    model = SpacyExtractor() if a.backend == 'spacy' else GLiNER2Extractor()
    result = []
    for i, row in enumerate(df.to_dict('records'), 1):
        # A headline without terminal punctuation can otherwise merge with the
        # first body sentence and create cross-boundary subject/object spans.
        texts = [row['title']] if a.source == 'title' else [row['title'], row['content']]
        events = [event for text in texts for event in model.extract(text, row['target_name'])]
        valid = [e for e in events if e['extraction_status'] == 'ok']
        if valid:
            events = valid
        result.append({**row, 'backend':a.backend, 'source':a.source,
                       'events_json':json.dumps(events,ensure_ascii=False),
                       'event_count':sum(e['extraction_status']=='ok' for e in events)})
        if i % 100 == 0:
            print('Processed:', i, '/', len(df), flush=True)
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(result).to_csv(path,index=False)
    print('Saved:',path, 'rows:',len(result))
    versions = {}
    for package in ('pandas', 'spacy', 'en-core-web-sm', 'gliner2', 'torch'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
        'backend': a.backend, 'source': a.source, 'rows': len(df),
        'model': 'en_core_web_sm' if a.backend == 'spacy' else 'fastino/gliner2-base-v1',
        'python': platform.python_version(), 'packages': versions,
        'missing_body_rows': int(df.content.str.strip().eq('').sum()),
        'full_text_policy': 'title and body parsed separately; repeated mentions are retained',
        'action_policy': 'spaCy contextual verb lemma where the extracted span is aligned',
        'note': 'Model revision is not pinned. Event extraction is not a stock forecast.',
    }
    path.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
if __name__=='__main__': main()
