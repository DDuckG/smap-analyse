from __future__ import annotations
from typing import Any
from smap.canonicalization.alias import normalize_alias
from smap.enrichers.models import EntityFact, IntentFact, SentimentFact, StanceFact
from smap.review.context import SemanticSignature
from smap.review.types import ReviewProblemClass

def build_entity_signature(entity_fact):
    normalized_candidate_text = normalize_alias(entity_fact.candidate_text) or None
    if entity_fact.unresolved_reason == 'ambiguous_alias':
        problem_class = ReviewProblemClass.AMBIGUOUS_ENTITY_MAPPING
    elif entity_fact.unresolved_reason == 'noise_term':
        problem_class = ReviewProblemClass.SUSPICIOUS_NOISE_CANDIDATE
    else:
        problem_class = ReviewProblemClass.UNRESOLVED_ENTITY_CANDIDATE
    ambiguity_signature = None
    if entity_fact.canonical_candidate_ids:
        ambiguity_signature = '|'.join(sorted(entity_fact.canonical_candidate_ids))
    return SemanticSignature(item_type='entity', problem_class=problem_class, normalized_candidate_text=normalized_candidate_text, entity_type_hint=entity_fact.entity_type, ambiguity_signature=ambiguity_signature, candidate_canonical_ids=sorted(entity_fact.canonical_candidate_ids))

def build_classification_signature(fact):
    if isinstance(fact, SentimentFact):
        label_key = f'sentiment:{fact.sentiment}'
    elif isinstance(fact, IntentFact):
        label_key = f'intent:{fact.intent}'
    else:
        label_key = f'stance:{fact.stance}'
    return SemanticSignature(item_type='classification', problem_class=ReviewProblemClass.LOW_CONFIDENCE_CLASSIFICATION, label_key=label_key)

def build_signature_from_payload(item_type, payload):
    if item_type == 'entity':
        candidate_text = payload.get('candidate_text')
        normalized_candidate_text = normalize_alias(candidate_text) if isinstance(candidate_text, str) else None
        canonical_ids_raw = payload.get('canonical_candidate_ids')
        candidate_canonical_ids = sorted((str(item) for item in canonical_ids_raw)) if isinstance(canonical_ids_raw, list) else []
        ambiguity_signature = '|'.join(candidate_canonical_ids) if candidate_canonical_ids else None
        unresolved_reason = payload.get('unresolved_reason')
        if unresolved_reason == 'ambiguous_alias':
            problem_class = ReviewProblemClass.AMBIGUOUS_ENTITY_MAPPING
        elif unresolved_reason == 'noise_term':
            problem_class = ReviewProblemClass.SUSPICIOUS_NOISE_CANDIDATE
        else:
            problem_class = ReviewProblemClass.UNRESOLVED_ENTITY_CANDIDATE
        return SemanticSignature(item_type='entity', problem_class=problem_class, normalized_candidate_text=normalized_candidate_text, entity_type_hint=str(payload.get('entity_type')) if payload.get('entity_type') else None, ambiguity_signature=ambiguity_signature, candidate_canonical_ids=candidate_canonical_ids)
    if 'sentiment' in payload:
        label_key = f"sentiment:{payload['sentiment']}"
    elif 'intent' in payload:
        label_key = f"intent:{payload['intent']}"
    else:
        label_key = f"stance:{payload.get('stance', 'unknown')}"
    return SemanticSignature(item_type='classification', problem_class=ReviewProblemClass.LOW_CONFIDENCE_CLASSIFICATION, label_key=label_key)
