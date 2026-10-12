"""Local and API-free syntactic event extraction baseline."""
import re
from dataclasses import dataclass, asdict

@dataclass
class Event:
    actor: str = ''
    action: str = ''
    object: str = ''
    event_type: str = 'other'
    target_role: str = 'unknown'
    extraction_status: str = 'ok'

def target_role(actor, obj, target_name, aliases=()):
    names = [target_name, *aliases]
    def present(value):
        return any(re.search(r'(?<!\w)' + re.escape(n.casefold()) + r'(?!\w)',
                             str(value).casefold()) for n in names if n)
    a, o = present(actor), present(obj)
    return 'both' if a and o else 'actor' if a else 'object' if o else ('other' if actor or obj else 'unknown')

def event_type(action, text):
    s = (str(action) + ' ' + str(text)).casefold()
    # Fixed heuristics, not a trained financial event classifier. Regulation
    # precedes product launch so "launch an investigation" is not a product.
    rules = {'litigation': [r'sue(?:s|d)?', r'suing', r'lawsuits?', r'courts?'],
             'earnings': [r'earnings', r'quarterly profits?', r'revenues?'],
             'merger_acquisition': [r'acquir(?:e|es|ed|ing)', r'mergers?', r'buyouts?'],
             'regulation': [r'regulators?', r'investigat(?:e|es|ed|ing|ion|ions)', r'antitrust'],
             'product_launch': [r'unveil(?:s|ed|ing)?', r'new products?', r'launch(?:es|ed|ing)?']}
    for category, patterns in rules.items():
        if any(re.search(r'(?<!\w)(?:' + pattern + r')(?!\w)', s) for pattern in patterns):
            return category
    return 'other'

class SpacyExtractor:
    def __init__(self, name='en_core_web_sm'):
        import spacy
        try:
            self.nlp = spacy.load(name)
        except OSError as exc:
            raise RuntimeError('Run: python -m spacy download en_core_web_sm') from exc

    def extract(self, text, target_name, aliases=()):
        if not str(text).strip():
            return [asdict(Event(extraction_status='empty_input'))]
        events = []
        for sent in self.nlp(text).sents:
            for verb in sent:
                if verb.pos_ not in ('VERB', 'AUX'):
                    continue
                children = list(verb.children)
                subjects = [c for c in children if c.dep_ in ('nsubj', 'csubj', 'nsubjpass')]
                objects = [c for c in children if c.dep_ in ('dobj', 'obj', 'dative', 'attr')]
                def phrase(tok):
                    return ' '.join(t.text for t in tok.subtree) if tok else ''
                subj = phrase(subjects[0]) if subjects else ''
                obj = phrase(objects[0]) if objects else ''
                passive = any(s.dep_ == 'nsubjpass' for s in subjects)
                agents = [z for c in children if c.dep_ == 'agent' for z in c.children if z.dep_ == 'pobj']
                actor, affected = (phrase(agents[0]) if agents else '', subj) if passive else (subj, obj)
                if not actor and not affected:
                    continue
                events.append(asdict(Event(actor=actor, action=verb.lemma_, object=affected,
                               event_type=event_type(verb.lemma_, sent.text),
                               target_role=target_role(actor, affected, target_name, aliases))))
        return events or [asdict(Event(extraction_status='no_event_detected'))]

class GLiNER2Extractor:
    def __init__(self, name='fastino/gliner2-base-v1'):
        from gliner2 import GLiNER2
        self.model = GLiNER2.from_pretrained(name)
        # GLiNER2 returns surface spans ("sues"), whereas spaCy returns lemmas
        # ("sue"). Use the same contextual lemmatizer for comparable actions.
        self.nlp = SpacyExtractor().nlp

    def extract(self, text, target_name, aliases=()):
        if not str(text).strip():
            return [asdict(Event(extraction_status='empty_input'))]
        schema = {'event': ['actor::str::Entity performing action',
                            'action::str::Main action verb',
                            'object::str::Affected entity']}
        raw = self.model.extract_json(text, schema)
        if not isinstance(raw, dict):
            raise ValueError('Unexpected GLiNER2 response; inspect installed version')
        items = raw.get('event', [])
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            raise ValueError('Unexpected GLiNER2 event collection')
        doc = self.nlp(text)
        events = []
        for item in items:
            if not isinstance(item, dict):
                continue
            a, v, o = (str(item.get(k) or '') for k in ('actor', 'action', 'object'))
            match = re.search(re.escape(v), text, flags=re.IGNORECASE) if v else None
            if match:
                span = doc.char_span(match.start(), match.end())
                if span is not None and span.root.pos_ in ('VERB', 'AUX'):
                    v = span.root.lemma_
            events.append(asdict(Event(actor=a, action=v, object=o,
                        target_role=target_role(a, o, target_name, aliases),
                        event_type=event_type(v, text))))
        return events or [asdict(Event(extraction_status='no_event_detected'))]
