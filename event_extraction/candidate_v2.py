"""Development-only candidate filter; not an independently validated extractor.

Separate product/brand references from the company's grammatical role. Preserve
negation/modality and incomplete roles instead of inventing missing entities.
This module is deliberately not wired into the frozen V1 forecasting pipeline.
"""
import re
from event_extraction.extractors import SpacyExtractor, event_type, target_role
from experiments.audit_apple_extraction import diagnose_segment, REPORTING

APPLE_BRANDS=re.compile(
    r'\b(?:Apple\s+(?:Watch|Pencil|Pay|Music|TV\+?|Podcasts|Arcade|Books|Fitness\+?|Business\s+Connect|Vision\s+Pro)|'
    r'AppleCare\+?|App\s+Store|iPhone|iPad|MacBook|HomePod|AirPods|iOS|iPadOS|macOS|watchOS)\b',re.I)
PRODUCT_WORDS=re.compile(
    r'\b(?:iphone|ipad|mac|macbook|homepod|airpods|watch|pencil|pay|business connect|vision pro|ios|ipados|macos|watchos|'
    r'products?|services?|software|apps?|features?|tools?|platforms?|chips?|resources?|collections?|diagnostics|video capture)\b',re.I)
PERSONNEL_WORDS=re.compile(r'\b(?:CEO|CFO|chief|officer|president|director)\b',re.I)
FINANCIAL_RESULTS=re.compile(r'\b(?:financial results|(?:first|second|third|fourth|[1-4](?:st|nd|rd|th)) quarter results|quarterly results)\b',re.I)


def company_reference(actor,obj,target='Apple'):
    """A known Apple-branded product is not the legal company as an object."""
    def direct(value):
        cleaned=APPLE_BRANDS.sub('',value)
        # Possessive names indicate ownership, not a direct company argument.
        cleaned=re.sub(r"\bApple[’']s\b",'',cleaned,flags=re.I)
        return cleaned
    role=target_role(direct(actor),direct(obj),target)
    has_brand=bool(APPLE_BRANDS.search(actor+' '+obj))
    if role=='unknown' and has_brand and (actor or obj):
        role='other'
    scope='direct_company' if role in {'actor','object','both'} else 'brand' if has_brand else 'none'
    return role,scope


def candidate_type(action,obj,sentence):
    old=event_type(action,sentence)
    if old in {'litigation','regulation','merger_acquisition'}:
        return old
    if action in {'report','announce','release','post','publish'} and FINANCIAL_RESULTS.search(sentence):
        return 'earnings'
    if action in {'introduce','unveil','launch','release'} and PRODUCT_WORDS.search(obj+' '+sentence) and not PERSONNEL_WORDS.search(obj):
        return 'product_launch'
    return old


def filter_candidates(records):
    kept=[]; rejected=[]; seen=set()
    for original in records:
        record=dict(original)
        reasons=[]
        if record['verb_pos']=='AUX': reasons.append('auxiliary_only')
        if record['action'].casefold() in REPORTING: reasons.append('reporting_or_mental_verb')
        if 'long_argument_span' in record['flags']: reasons.append('long_argument_span')
        if 'target_only_in_argument_subtree' in record['flags']: reasons.append('nested_target_requires_review')
        if not record['actor'] and record.get('conj_actor'):
            record['actor']=record['conj_actor']
            record['actor_inherited_from_conj']=True
        else:
            record['actor_inherited_from_conj']=False
        role,scope=company_reference(record['actor'],record['object'])
        if scope=='none': reasons.append('no_direct_company_or_known_brand_argument')
        record['v1_target_role']=record['target_role']
        record['target_role']=role
        record['target_reference']=scope
        record['v1_event_type']=record['event_type']
        record['event_type']=candidate_type(record['action'],record['object'],record['sentence'])
        record['extraction_status']='candidate' if record['actor'] and record['object'] else 'partial_requires_review'
        key=tuple(' '.join(str(record[k]).casefold().split()) for k in ('actor','action','object','event_type','target_role'))
        if key in seen: reasons.append('exact_duplicate')
        if reasons:
            rejected.append({**record,'rejection_reasons':reasons})
        else:
            seen.add(key)
            kept.append(record)
    return kept,rejected


class CandidateV2Extractor:
    def __init__(self):
        self.nlp=SpacyExtractor().nlp

    def extract(self,title,body=''):
        records=diagnose_segment(self.nlp,title)+diagnose_segment(self.nlp,body)
        return filter_candidates(records)
