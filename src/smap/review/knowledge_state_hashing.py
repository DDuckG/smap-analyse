from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import cast

from smap.ontology.models import (
    AliasContribution,
    EntitySeed,
    NoiseTerm,
    OntologyRegistry,
    OverlayProvenanceValue,
    TaxonomyNode,
)

_NORMALIZE_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def stable_content_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def ontology_semantic_fingerprint(registry: OntologyRegistry) -> str:
    payload = {
        "entities": _sorted_payload(_canonical_entity(entity) for entity in registry.entities),
        "entity_types": sorted(registry.entity_types),
        "taxonomy_nodes": _sorted_payload(_canonical_taxonomy_node(node) for node in registry.taxonomy_nodes),
    }
    return stable_content_hash(payload)


def non_review_overlay_fingerprint(registry: OntologyRegistry) -> str:
    payload = {
        "alias_contributions": _sorted_payload(
            _canonical_alias_contribution(contribution)
            for contribution in registry.alias_contributions
            if contribution.source != "review"
        ),
        "noise_terms": _sorted_payload(
            _canonical_noise_term(noise_term)
            for noise_term in registry.noise_terms
            if noise_term.source != "review"
        ),
    }
    return stable_content_hash(payload)


def reviewed_overlay_fingerprint(registry: OntologyRegistry) -> str:
    reviewed_aliases = _sorted_payload(
        _canonical_alias_contribution(contribution)
        for contribution in registry.alias_contributions
        if contribution.source == "review"
    )
    reviewed_noise_terms = _sorted_payload(
        _canonical_noise_term(noise_term)
        for noise_term in registry.noise_terms
        if noise_term.source == "review"
    )
    if not reviewed_aliases and not reviewed_noise_terms:
        return "none"
    return stable_content_hash(
        {
            "alias_contributions": reviewed_aliases,
            "noise_terms": reviewed_noise_terms,
        }
    )


def _canonical_entity(entity: EntitySeed) -> dict[str, object]:
    return {
        "id": entity.id,
        "name": entity.name.casefold(),
        "entity_type": entity.entity_type,
        "entity_kind": entity.entity_kind,
        "knowledge_layer": entity.knowledge_layer,
        "active_linking": entity.active_linking,
        "aliases": sorted({_normalize_surface(alias) or alias.casefold() for alias in entity.aliases}),
        "taxonomy_ids": sorted(entity.taxonomy_ids),
    }


def _canonical_taxonomy_node(node: TaxonomyNode) -> dict[str, object]:
    return {
        "id": node.id,
        "label": node.label.casefold(),
        "node_type": node.node_type,
        "parent_id": node.parent_id,
    }


def _canonical_alias_contribution(contribution: AliasContribution) -> dict[str, object]:
    return {
        "canonical_entity_id": contribution.canonical_entity_id,
        "normalized_alias": _normalize_surface(contribution.alias) or contribution.alias.casefold(),
        "entity_type": contribution.entity_type,
        "source": contribution.source,
        "semantic_provenance": _canonical_semantic_provenance(contribution.provenance),
    }


def _canonical_noise_term(noise_term: NoiseTerm) -> dict[str, object]:
    return {
        "normalized_term": _normalize_surface(noise_term.term) or noise_term.term.casefold(),
        "source": noise_term.source,
        "semantic_provenance": _canonical_semantic_provenance(noise_term.provenance),
    }


def _canonical_semantic_provenance(
    provenance: Mapping[str, OverlayProvenanceValue],
) -> dict[str, object]:
    applicability_policy = _load_json_object(provenance.get("applicability_policy_json"))
    scope_key = _load_json_object(provenance.get("scope_key_json"))
    semantic_signature = _load_json_object(provenance.get("semantic_signature_json"))

    canonical_payload: dict[str, object] = {}
    if applicability_policy is not None:
        canonical_payload["applicability_policy"] = {
            "applies_to_problem_class": applicability_policy.get("applies_to_problem_class"),
            "authority_level": applicability_policy.get("authority_level"),
            "future_effect": applicability_policy.get("future_effect"),
            "terminates_future_authority": applicability_policy.get("terminates_future_authority"),
            "match_mode": applicability_policy.get("match_mode"),
            "knowledge_match_mode": applicability_policy.get("knowledge_match_mode"),
            "valid_signature_version": applicability_policy.get("valid_signature_version"),
            "valid_scope_key": applicability_policy.get("valid_scope_key"),
            "semantic_match_value": applicability_policy.get("semantic_match_value"),
            "semantic_fingerprint": applicability_policy.get("semantic_fingerprint"),
            "origin_knowledge_state": applicability_policy.get("origin_knowledge_state"),
        }
    if scope_key is not None:
        canonical_payload["scope_key"] = scope_key
    if semantic_signature is not None:
        canonical_payload["semantic_signature"] = {
            "item_type": semantic_signature.get("item_type"),
            "problem_class": semantic_signature.get("problem_class"),
            "normalized_candidate_text": semantic_signature.get("normalized_candidate_text"),
            "entity_type_hint": semantic_signature.get("entity_type_hint"),
            "ambiguity_signature": semantic_signature.get("ambiguity_signature"),
            "candidate_canonical_ids": semantic_signature.get("candidate_canonical_ids", []),
            "label_key": semantic_signature.get("label_key"),
            "signature_version": semantic_signature.get("signature_version"),
        }
    return canonical_payload


def canonical_review_alias_contribution(contribution: AliasContribution) -> dict[str, object]:
    return _canonical_alias_contribution(contribution)


def canonical_review_noise_term(noise_term: NoiseTerm) -> dict[str, object]:
    return _canonical_noise_term(noise_term)


def canonical_semantic_provenance(
    provenance: Mapping[str, OverlayProvenanceValue],
) -> dict[str, object]:
    return _canonical_semantic_provenance(provenance)


def _sorted_payload(items: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    payload_items = list(items)
    return sorted(payload_items, key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True))


def _normalize_surface(text: str) -> str:
    normalized = _NORMALIZE_RE.sub(" ", text.lower())
    return " ".join(normalized.split())


def _load_json_object(value: OverlayProvenanceValue) -> dict[str, object] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        return None
    return cast(dict[str, object], loaded)
