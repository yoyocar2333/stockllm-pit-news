from event_extraction.extractors import target_role, event_type, Event

def test_company_role():
    assert target_role('Apple','Samsung','Apple') == 'actor'
    assert target_role('Samsung','Apple','Apple') == 'object'
    assert target_role('Apple','Apple','Apple') == 'both'
    assert target_role('Pineapple','Samsung','Apple') == 'other'
    assert target_role('','','Apple') == 'unknown'

def test_categories():
    assert event_type('sue','Apple sues Samsung') == 'litigation'
    assert event_type('report','quarterly earnings') == 'earnings'

def test_missing_case():
    assert Event(extraction_status='no_event_detected').target_role == 'unknown'

def test_investigation_is_not_product_launch():
    assert event_type('launch', 'The regulators launched a new investigation.') == 'regulation'
    assert event_type('investigate', 'Regulators investigate Apple') == 'regulation'

def test_passive_agent_and_patient():
    import pytest
    pytest.importorskip('en_core_web_sm')
    from event_extraction.extractors import SpacyExtractor
    events = SpacyExtractor().extract('Apple was sued by Samsung.', 'Apple')
    event = next(e for e in events if e['action'] == 'sue')
    assert event['actor'] == 'Samsung'
    assert event['object'] == 'Apple'
    assert event['target_role'] == 'object'
