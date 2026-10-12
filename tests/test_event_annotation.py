import json
import pandas as pd
import pytest
from event_extraction.annotate import FIELDS, prepare, score


@pytest.fixture
def review(tmp_path):
    event = dict(actor='Apple', action='sue', object='Samsung', event_type='litigation',
                 target_role='actor', extraction_status='ok')
    predictions = tmp_path / 'predictions.csv'
    labels = tmp_path / 'labels.csv'
    pd.DataFrame([dict(article_id='0001', ticker='AAPL', target_name='Apple',
                       title='Apple sues Samsung.', content='Hidden body.', source='title',
                       events_json=json.dumps([event]))]).to_csv(predictions, index=False)
    prepare(predictions, labels)
    df = pd.read_csv(labels, dtype=str, keep_default_na=False)
    df.loc[0, 'review_status'] = 'reviewed'
    df.loc[0, 'decision'] = 'keep'
    df.loc[0, 'evidence'] = 'Apple sues Samsung.'
    for field in FIELDS:
        df.loc[0, 'gold_' + field] = event[field]
    return predictions, labels, df


def test_scores_complete_reference_and_preserves_ids(review):
    predictions, labels, df = review
    assert df.loc[0, 'article_id'] == '0001'
    assert df.loc[0, 'content'] == ''
    df.to_csv(labels, index=False)
    result = score(predictions, labels)
    assert result['triple']['f1'] == 1
    assert result['scored_articles'] == 1


def test_missing_gold_event_counts_as_false_negative(review):
    predictions, labels, df = review
    extra = df.iloc[0].copy()
    extra['row_id'] = '0001:added1'
    extra['decision'] = 'add'
    extra['gold_action'] = 'announce'
    df = pd.concat([df, pd.DataFrame([extra])], ignore_index=True)
    df.to_csv(labels, index=False)
    result = score(predictions, labels)
    assert result['triple']['false_negative'] == 1
    assert result['triple']['recall'] == .5


def test_delete_is_false_positive(review):
    predictions, labels, df = review
    df.loc[0, 'decision'] = 'delete'
    for field in FIELDS:
        df.loc[0, 'gold_' + field] = ''
    df.to_csv(labels, index=False)
    result = score(predictions, labels)
    assert result['triple']['false_positive'] == 1
    assert result['triple']['f1'] == 0
    assert result['triple']['recall'] is None


@pytest.mark.parametrize('change,match', [('pending', 'Finish'), ('wrong_evidence', 'Evidence'),
                                        ('wrong_source', 'source'), ('drop_article', 'row_id')])
def test_rejects_invalid_annotations(review, change, match):
    predictions, labels, df = review
    if change == 'pending':
        df.loc[0, 'review_status'] = 'pending'
    elif change == 'wrong_evidence':
        df.loc[0, 'evidence'] = 'Made up evidence'
    elif change == 'wrong_source':
        df.loc[0, 'source'] = 'full'
    else:
        df = df.iloc[:0]
    df.to_csv(labels, index=False)
    with pytest.raises(ValueError, match=match):
        score(predictions, labels)


def test_blind_export_hides_predictions_and_protects_work(review):
    predictions, labels, _ = review
    blind = labels.with_name('blind.csv')
    prepare(predictions, blind, blind=True)
    df = pd.read_csv(blind, dtype=str, keep_default_na=False)
    assert df.loc[0, 'annotation_mode'] == 'blind'
    assert all(df.loc[0, 'predicted_' + field] == '' for field in FIELDS)
    with pytest.raises(ValueError, match='overwrite'):
        prepare(predictions, blind, blind=True)
