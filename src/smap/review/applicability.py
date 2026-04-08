from __future__ import annotations

from smap.ontology.models import OverlayProvenanceValue
from smap.review.applicability_engine import ApplicabilityEngine
from smap.review.context import ReviewContext, ScopeKey
from smap.review.policy import DecisionApplicabilityPolicy

_ENGINE = ApplicabilityEngine()


def scope_key_to_provenance(scope_key: ScopeKey) -> dict[str, OverlayProvenanceValue]:
    return _ENGINE.scope_key_to_provenance(scope_key)


def applicability_to_provenance(
    policy: DecisionApplicabilityPolicy,
) -> dict[str, OverlayProvenanceValue]:
    return _ENGINE.applicability_to_provenance(policy)


def scope_key_from_provenance(provenance: dict[str, OverlayProvenanceValue]) -> ScopeKey | None:
    return _ENGINE.scope_key_from_provenance(provenance)


def applicability_matches_context(
    provenance: dict[str, OverlayProvenanceValue],
    context: ReviewContext | None,
) -> bool:
    return _ENGINE.contribution_applies(provenance, context)
