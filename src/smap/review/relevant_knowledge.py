from __future__ import annotations
import json
import re
from collections.abc import Mapping
from typing import cast
from pydantic import BaseModel
from smap.ontology.models import AliasContribution, NoiseTerm, OntologyRegistry, OverlayProvenanceValue
from smap.review.context import REVIEW_SIGNATURE_VERSION, ReviewContext, SemanticSignature
from smap.review.knowledge_state_hashing import canonical_review_alias_contribution, canonical_review_noise_term, stable_content_hash
from smap.review.types import ReviewProblemClass
_NORMALIZE_RE = re.compile('[^\\w\\s]', flags=re.UNICODE)

class RelevantSemanticRegion(BaseModel):
    problem_class: ReviewProblemClass
    normalized_candidate_text: str | None = None
    entity_type_hint: str | None = None
    ambiguity_signature: str | None = None
    signature_version: int = REVIEW_SIGNATURE_VERSION

    @classmethod
    def from_signature(cls, signature):
        return cls(problem_class=signature.problem_class, normalized_candidate_text=signature.normalized_candidate_text, entity_type_hint=signature.entity_type_hint, ambiguity_signature=signature.ambiguity_signature, signature_version=signature.signature_version)

    @property
    def is_entity_like(self):
        return self.problem_class in {ReviewProblemClass.UNRESOLVED_ENTITY_CANDIDATE, ReviewProblemClass.AMBIGUOUS_ENTITY_MAPPING, ReviewProblemClass.SUSPICIOUS_NOISE_CANDIDATE}

def relevant_reviewed_knowledge_fingerprint(registry, *, signature, context):
    region = RelevantSemanticRegion.from_signature(signature)
    if not region.is_entity_like or not region.normalized_candidate_text:
        return 'none'
    relevant_aliases = _sorted_payload((canonical_review_alias_contribution(contribution) for contribution in registry.alias_contributions if contribution.source == 'review' and _alias_is_relevant(contribution, region, context)))
    relevant_noise_terms = _sorted_payload((canonical_review_noise_term(noise_term) for noise_term in registry.noise_terms if noise_term.source == 'review' and _noise_term_is_relevant(noise_term, region, context)))
    if not relevant_aliases and (not relevant_noise_terms):
        return 'none'
    return stable_content_hash({'semantic_region': region.model_dump(mode='json'), 'alias_contributions': relevant_aliases, 'noise_terms': relevant_noise_terms})

def _alias_is_relevant(contribution, region, context):
    if _normalize_surface(contribution.alias) != region.normalized_candidate_text:
        return False
    if region.entity_type_hint is not None and contribution.entity_type is not None and (contribution.entity_type != region.entity_type_hint):
        return False
    if not _scope_matches_context(contribution.provenance, context):
        return False
    return _base_knowledge_matches_context(contribution.provenance, context)

def _noise_term_is_relevant(noise_term, region, context):
    if _normalize_surface(noise_term.term) != region.normalized_candidate_text:
        return False
    if not _scope_matches_context(noise_term.provenance, context):
        return False
    return _base_knowledge_matches_context(noise_term.provenance, context)

def _scope_matches_context(provenance, context):
    applicability_policy = _load_json_object(provenance.get('applicability_policy_json'))
    if applicability_policy is not None:
        scope_key = applicability_policy.get('valid_scope_key')
        if isinstance(scope_key, dict):
            return _static_scope_matches(scope_key.get('static_scope'), context)
    scope_key = _load_json_object(provenance.get('scope_key_json'))
    if scope_key is None:
        return True
    return _static_scope_matches(scope_key.get('static_scope'), context)

def _static_scope_matches(scope_payload, context):
    if not isinstance(scope_payload, dict):
        return True
    for field_name in ('project_id', 'task_id', 'platform', 'root_id', 'language_family'):
        expected_value = scope_payload.get(field_name)
        if expected_value is not None and getattr(context, field_name) != expected_value:
            return False
    return True

def _base_knowledge_matches_context(provenance, context):
    applicability_policy = _load_json_object(provenance.get('applicability_policy_json'))
    if applicability_policy is not None:
        origin_state = applicability_policy.get('origin_knowledge_state')
        if isinstance(origin_state, dict):
            return _knowledge_state_matches(origin_state, context)
    scope_key = _load_json_object(provenance.get('scope_key_json'))
    if scope_key is None:
        return True
    knowledge_state = scope_key.get('knowledge_state')
    if not isinstance(knowledge_state, dict):
        return True
    return _knowledge_state_matches(knowledge_state, context)

def _knowledge_state_matches(knowledge_state, context):
    ontology_fingerprint = knowledge_state.get('ontology_fingerprint')
    overlay_fingerprint = knowledge_state.get('overlay_fingerprint')
    if ontology_fingerprint not in {None, ''} and ontology_fingerprint != context.ontology_fingerprint:
        return False
    return overlay_fingerprint in {None, ''} or overlay_fingerprint == context.overlay_fingerprint

def _sorted_payload(items):
    payload_items = list(cast(list[dict[str, object]], items))
    return sorted(payload_items, key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True))

def _normalize_surface(text):
    normalized = _NORMALIZE_RE.sub(' ', text.lower())
    return ' '.join(normalized.split())

def _load_json_object(value):
    if not isinstance(value, str) or not value.strip():
        return None
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        return None
    return cast(dict[str, object], loaded)
