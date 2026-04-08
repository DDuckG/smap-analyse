from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from smap.canonicalization.alias import AliasRegistry, normalize_alias
from smap.core.exceptions import ReviewConflictError, ReviewValidationError
from smap.core.settings import Settings
from smap.ontology.loader import load_ontology, load_ontology_overlay
from smap.ontology.models import AliasContribution, NoiseTerm, OntologyOverlay
from smap.ontology.runtime import load_runtime_ontology
from smap.review.applicability_engine import ApplicabilityEngine
from smap.review.policy import DecisionApplicabilityPolicy
from smap.review.subsumption import ContributionSubsumptionEngine
from smap.review.types import ContributionAction


def _load_review_overlay(settings: Settings) -> OntologyOverlay | None:
    if not settings.review_overlay_path.exists():
        return None
    return load_ontology_overlay(settings.review_overlay_path)


def validate_remap_target(settings: Settings, remap_target: str) -> None:
    registry = load_runtime_ontology(settings).registry
    entity_ids = {entity.id for entity in registry.entities}
    if remap_target not in entity_ids:
        raise ReviewValidationError(f"Unknown canonical entity target: {remap_target}")


def validate_alias_contribution(
    settings: Settings,
    *,
    aliases: Sequence[str],
    remap_target: str,
    applicability_policy: DecisionApplicabilityPolicy,
    allow_replacement: bool = False,
) -> None:
    validate_remap_target(settings, remap_target)
    registry = load_runtime_ontology(settings).registry
    alias_registry = AliasRegistry.from_ontology(registry)
    review_overlay = _load_review_overlay(settings)
    engine = ApplicabilityEngine()
    subsumption_engine = ContributionSubsumptionEngine(engine)
    for alias_text in aliases:
        normalized_alias = normalize_alias(alias_text)
        if not normalized_alias:
            raise ReviewValidationError("Alias contribution requires non-empty alias text.")
        if alias_registry.is_noise_term(normalized_alias):
            raise ReviewConflictError(f"Alias `{alias_text}` is currently suppressed as noise.")
        existing_aliases = alias_registry.aliases_by_normalized.get(normalized_alias, [])
        existing_targets = {alias.canonical_entity_id for alias in existing_aliases}
        if existing_targets and existing_targets != {remap_target}:
            raise ReviewConflictError(
                f"Alias `{alias_text}` already maps to {sorted(existing_targets)}; refusing remap to {remap_target}."
            )
        if review_overlay is None:
            continue
        relevant_contributions: list[AliasContribution | NoiseTerm] = [
            contribution
            for contribution in review_overlay.alias_contributions
            if normalize_alias(contribution.alias) == normalized_alias
        ]
        relevant_contributions.extend(
            noise_term
            for noise_term in review_overlay.noise_terms
            if normalize_alias(noise_term.term) == normalized_alias
        )
        reconciliation = subsumption_engine.reconcile_contributions(
            existing_contributions=relevant_contributions,
            incoming_policy=applicability_policy,
            incoming_target=remap_target,
            incoming_kind="alias",
            allow_authoritative_replacement=allow_replacement,
        )
        if reconciliation.action == ContributionAction.REJECT_CONFLICT:
            raise ReviewConflictError(
                f"Alias `{alias_text}` has conflicting reviewed knowledge; {reconciliation.reason}."
            )


def validate_noise_contribution(
    settings: Settings,
    *,
    aliases: Sequence[str],
    applicability_policy: DecisionApplicabilityPolicy,
    allow_replacement: bool = False,
) -> None:
    registry = load_runtime_ontology(settings).registry
    alias_registry = AliasRegistry.from_ontology(registry)
    review_overlay = _load_review_overlay(settings)
    engine = ApplicabilityEngine()
    subsumption_engine = ContributionSubsumptionEngine(engine)
    for alias_text in aliases:
        normalized_alias = normalize_alias(alias_text)
        if not normalized_alias:
            raise ReviewValidationError("Noise contribution requires non-empty alias text.")
        existing_aliases = alias_registry.aliases_by_normalized.get(normalized_alias, [])
        if existing_aliases:
            existing_targets = sorted({alias.canonical_entity_id for alias in existing_aliases})
            raise ReviewConflictError(
                f"Alias `{alias_text}` is an active ontology/overlay alias for {existing_targets}; refusing noise suppression."
            )
        if review_overlay is None:
            continue
        relevant_contributions: list[AliasContribution | NoiseTerm] = [
            contribution
            for contribution in review_overlay.alias_contributions
            if normalize_alias(contribution.alias) == normalized_alias
        ]
        relevant_contributions.extend(
            noise_term
            for noise_term in review_overlay.noise_terms
            if normalize_alias(noise_term.term) == normalized_alias
        )
        reconciliation = subsumption_engine.reconcile_contributions(
            existing_contributions=relevant_contributions,
            incoming_policy=applicability_policy,
            incoming_target=None,
            incoming_kind="noise",
            allow_authoritative_replacement=allow_replacement,
        )
        if reconciliation.action == ContributionAction.REJECT_CONFLICT:
            raise ReviewConflictError(
                f"Alias `{alias_text}` already has incompatible reviewed knowledge; {reconciliation.reason}."
            )


def validate_overlay_reload(settings: Settings, *, overlay_path: Path | None = None) -> None:
    overlay_paths: list[Path] = []
    if overlay_path is not None:
        overlay_paths.append(overlay_path)
    elif settings.review_overlay_path.exists():
        overlay_paths.append(settings.review_overlay_path)
    load_ontology(settings.selected_domain_ontology_path, overlay_paths=overlay_paths)
