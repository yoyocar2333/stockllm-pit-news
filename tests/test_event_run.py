import hashlib
import json
import sys
import pandas as pd
import pytest


def test_full_cli_keeps_headline_and_body_boundaries(tmp_path, monkeypatch):
    pytest.importorskip('en_core_web_sm')
    from event_extraction.run import main
    input_path = tmp_path / 'news.csv'
    output = tmp_path / 'events.csv'
    pd.DataFrame([dict(article_id='0001', ticker='TSLA', target_name='Tesla',
                       title='Tesla unveils new car prototype',
                       content='Tesla introduced a new electric vehicle prototype.')]).to_csv(input_path, index=False)
    monkeypatch.setattr(sys, 'argv', ['run', '--input', str(input_path), '--output', str(output),
                                     '--backend', 'spacy', '--source', 'full'])
    main()
    rows = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert rows.iloc[0].article_id == '0001'
    events = json.loads(rows.iloc[0].events_json)
    assert {event['action'] for event in events} == {'unveil', 'introduce'}
    assert all(event['actor'] == 'Tesla' and event['object'] for event in events)
    metadata = json.loads(output.with_suffix('.meta.json').read_text(encoding='utf-8'))
    assert metadata['input_sha256'] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert metadata['source'] == 'full' and metadata['rows'] == 1
