import json
import pytest

from experiments.review_extraction_gate import (
    POLICY, compare_backend, load_ai_draft, rejection_risks, sha256,
    validate_events, write_blind_page,
)
from experiments.score_independent_extraction import load_reference


def event(action='launch'):
    return dict(actor='Apple', action=action, object='a product',
                event_type='product_launch', target_role='actor')


def draft(article_id='a'):
    return dict(article_id=article_id, url='https://example.com/' + article_id,
                source='title', reviewer_type='ai_assisted', reviewer_id='assistant',
                review_status='reviewed', no_event=False, events=[event()],
                title_sha256=sha256(b'Apple launches a product'),
                source_input_sha256='source-sha', annotation_policy=POLICY)


def test_ai_draft_requires_same_source_and_explicit_provenance(tmp_path):
    row = draft()
    article = dict(article_id='a', url=row['url'], title='Apple launches a product')
    freeze = dict(source_input_sha256='source-sha')
    path = tmp_path / 'draft.jsonl'
    path.write_text(json.dumps(row))
    assert load_ai_draft(path, freeze, [article])['a']['events'] == [event()]
    row['reviewer_type'] = 'human_independent'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='AI-assisted'):
        load_ai_draft(path, freeze, [article])
    row = draft(); row['title_sha256'] = 'changed'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='snapshot'):
        load_ai_draft(path, freeze, [article])
    row = draft(); row['source_input_sha256'] = 'changed'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='source hash'):
        load_ai_draft(path, freeze, [article])


def test_ai_draft_rejects_duplicate_ids_and_empty_actions(tmp_path):
    row = draft(); path = tmp_path / 'draft.jsonl'
    path.write_text(json.dumps(row) + '\n' + json.dumps(row))
    articles = [dict(article_id='a'), dict(article_id='b')]
    with pytest.raises(ValueError, match='IDs'):
        load_ai_draft(path, dict(source_input_sha256='source-sha'), articles)
    row['events'][0]['action'] = ''
    with pytest.raises(ValueError, match='Action'):
        validate_events(row)
    row = draft(); row['no_event'] = True
    with pytest.raises(ValueError, match='disagree'):
        validate_events(row)


def test_action_agreement_is_not_five_field_agreement():
    pred = event(); pred['object'] = 'different product'
    totals, _ = compare_backend({'a': [pred]}, {'a': draft()}, [dict(article_id='a', url='url')])
    assert totals['shared_action_occurrences'] == 1
    assert totals['strict_shared_records'] == 0
    assert totals['exact_article_agreement'] == 0


def test_empty_outputs_and_rejected_action_risk_are_visible():
    article = dict(article_id='a', url='url')
    totals, _ = compare_backend({'a': []}, {'a': draft()}, [article])
    assert totals['zero_output_with_nonempty_ai_draft'] == 1
    rejected = dict(action='launch', rejection_reasons=['unrecognized_brand'])
    risks = rejection_risks({'a': dict(kept=[], rejected=[rejected])}, {'a': draft()}, [article])
    assert len(risks) == 1 and risks[0]['action'] == 'launch'
    with pytest.raises(ValueError, match='Missing backend'):
        compare_backend({}, {'a': draft()}, [article])


def test_declared_independent_but_ai_exposed_is_rejected(tmp_path):
    row = draft(); row.update(reviewer_type='human_independent', ai_exposed=True)
    path = tmp_path / 'gold.jsonl'; path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='AI-exposed'):
        load_reference(path, ['a'])


def test_blind_page_contains_titles_not_predictions_and_escapes_script(tmp_path):
    path = tmp_path / 'review.html'
    article = dict(article_id='a', title='</script><script>alert(1)</script>',
                   url='https://example.com/a', publication_date='2023-01-01')
    write_blind_page([article], dict(source_input_sha256='source-sha'), path)
    text = path.read_text(encoding='utf-8')
    assert '/*__REVIEW_PAYLOAD__*/ null' not in text
    assert '</script><script>alert(1)' not in text
    assert '\\u003c/script>' in text
    payload = json.loads(text.split('const DATA = ', 1)[1].split(';\n', 1)[0])
    assert payload['items'][0]['title'] == article['title']
    assert all('events' not in r and 'predictions' not in r for r in payload['items'])
