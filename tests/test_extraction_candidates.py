import pytest
from event_extraction.candidate_v2 import company_reference, candidate_type, filter_candidates, CandidateV2Extractor
from experiments.audit_apple_extraction import diagnose_segment


def test_company_is_not_its_product_object():
    assert company_reference('Apple','new Apple Watch')==('actor','direct_company')
    assert company_reference('Samsung','Apple')==('object','direct_company')
    assert company_reference('Apple Music','new playlist')==('other','brand')
    assert company_reference('Apple','Apple')==('both','direct_company')
    assert company_reference('Pineapple','pear')==('other','none')


def test_owned_entity_does_not_become_direct_company():
    assert company_reference("Apple's engineers",'a circuit')[0]=='other'


def test_financial_results_and_product_introduction_types():
    assert candidate_type('report','first quarter results','Apple reports first quarter results')=='earnings'
    assert candidate_type('introduce','new tablet','Apple introduces a new product')=='product_launch'
    assert candidate_type('introduce','new chief executive','Apple introduces its new chief executive')=='other'
    assert candidate_type('launch','an investigation','Regulators launch an investigation')=='regulation'


@pytest.fixture
def extractor():
    pytest.importorskip('en_core_web_sm')
    return CandidateV2Extractor()


def test_direct_negation_and_modality_are_preserved(extractor):
    negated,_=extractor.extract('Apple did not launch a new product.')
    assert next(e for e in negated if e['action']=='launch')['negated'] is True
    planned,_=extractor.extract('Apple may launch a new product.')
    assert 'may' in next(e for e in planned if e['action']=='launch')['modals']


def test_coordinated_verb_inherits_explicit_subject(extractor):
    kept,_=extractor.extract('Apple expands services and introduces a new tool.')
    introduced=next(e for e in kept if e['action']=='introduce')
    assert introduced['actor']=='Apple' and introduced['actor_inherited_from_conj'] is True


def test_rejected_events_still_available_for_review(extractor):
    kept,rejected=extractor.extract('Apple says the new product is available.')
    assert all(e['action']!='say' for e in kept)
    assert any('reporting_or_mental_verb' in e['rejection_reasons'] for e in rejected)


def test_exact_duplicates_are_not_silently_scored_as_new_events(extractor):
    records=diagnose_segment(extractor.nlp,'Apple launches a new product.')
    kept,rejected=filter_candidates(records+records)
    assert len(kept)==1
    assert any('exact_duplicate' in e['rejection_reasons'] for e in rejected)


def test_missing_subject_remains_partial(extractor):
    kept,_=extractor.extract('Introducing Apple Vision Pro.')
    event=next(e for e in kept if e['action']=='introduce')
    assert event['actor']=='' and event['extraction_status']=='partial_requires_review'
    assert event['target_role']=='other' and event['target_reference']=='brand'


def test_independent_scoring_rejects_pending_and_ai_labels(tmp_path):
    import json
    from experiments.score_independent_extraction import load_reference
    path=tmp_path/'gold.jsonl'
    row={'article_id':'a','source':'title','review_status':'pending','reviewer_type':'',
         'reviewer_id':'','no_event':None,'events':[]}
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='pending'):
        load_reference(path,['a'])
    row.update(review_status='reviewed',reviewer_type='ai_assisted',reviewer_id='assistant',no_event=True)
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='human reviewer'):
        load_reference(path,['a'])


def test_independent_scoring_counts_false_positives_and_missing_ids(tmp_path):
    import json
    from experiments.score_independent_extraction import load_reference, score_events
    row={'article_id':'a','source':'title','review_status':'reviewed','reviewer_type':'human_independent',
         'reviewer_id':'reviewer_1','no_event':True,'events':[]}
    path=tmp_path/'gold.jsonl'; path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='IDs'):
        load_reference(path,['a','b'])
    event={'actor':'Apple','action':'launch','object':'a product','event_type':'product_launch','target_role':'actor'}
    score=score_events([event,event],[event])
    assert score=={'tp':1,'predicted':2,'gold':1}
