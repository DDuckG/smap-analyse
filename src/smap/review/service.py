from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import Select, select
from sqlalchemy.orm import Session, selectinload

from smap.canonicalization.alias import normalize_alias
from smap.core.settings import Settings
from smap.core.types import utc_now
from smap.enrichers.models import EnrichmentBundle, IntentFact, SentimentFact, StanceFact
from smap.ontology.models import (
    AliasContribution,
    NoiseTerm,
    OntologyOverlay,
    OntologyOverlayMetadata,
    OntologyRegistry,
)
from smap.review.applicability import applicability_to_provenance
from smap.review.applicability_engine import ApplicabilityEngine
from smap.review.context import (
    ReviewContext,
    ScopeKey,
    SemanticSignature,
    build_ontology_fingerprint,
    build_overlay_fingerprint,
    build_reviewed_overlay_fingerprint,
)
from smap.review.grouping import (
    OPEN_REVIEW_STATUSES,
    attach_review_item_to_group,
    count_open_review_groups,
    effective_future_group_decision,
    ensure_review_group,
    get_group_context,
    get_group_scope_key,
    get_group_signature,
    get_review_group,
    has_applicable_future_decision,
    iter_group_candidate_texts,
    list_review_groups,
    register_suppressed_occurrence,
)
from smap.review.metrics import ReviewQueueSummary
from smap.review.models import (
    OntologyRegistryVersion,
    ReviewDecision,
    ReviewGroup,
    ReviewGroupDecision,
    ReviewItem,
)
from smap.review.policy import DecisionApplicabilityPolicy, ReviewPolicyEngine
from smap.review.schemas import ReviewDecisionCreate, ReviewGroupDecisionCreate
from smap.review.signatures import (
    build_classification_signature,
    build_entity_signature,
    build_signature_from_payload,
)
from smap.review.subsumption import ContributionSubsumptionEngine
from smap.review.types import (
    AuthorityLevel,
    ContributionAction,
    FutureEffectType,
    KnowledgeMatchMode,
    ReviewAction,
    ReviewResolutionScope,
    ReviewStatus,
)
from smap.review.validation import (
    validate_alias_contribution,
    validate_noise_contribution,
    validate_overlay_reload,
    validate_remap_target,
)


def seed_ontology_version(session: Session, registry: OntologyRegistry) -> None:
    existing = session.get(OntologyRegistryVersion, registry.metadata.version)
    if existing is None:
        session.add(
            OntologyRegistryVersion(
                version_id=registry.metadata.version,
                name=registry.metadata.name,
                description=registry.metadata.description,
            )
        )


def _read_review_overlay(path: Path) -> OntologyOverlay:
    if not path.exists():
        return OntologyOverlay(
            metadata=OntologyOverlayMetadata(
                name="reviewed-aliases",
                version=utc_now().isoformat(),
                description="Alias and noise contributions derived from review decisions.",
                source="review",
                layer_kind="review_overlay",
            )
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return OntologyOverlay.model_validate(payload)


def _write_review_overlay(settings: Settings, overlay: OntologyOverlay) -> None:
    settings.review_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = settings.review_overlay_path.with_suffix(".candidate.yaml")
    candidate_path.write_text(
        yaml.safe_dump(overlay.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    try:
        validate_overlay_reload(settings, overlay_path=candidate_path)
        candidate_path.replace(settings.review_overlay_path)
    finally:
        if candidate_path.exists():
            candidate_path.unlink()


def _unique_alias_texts(items: list[ReviewItem]) -> list[str]:
    unique_aliases: list[str] = []
    seen: set[str] = set()
    for item in items:
        candidate_text = item.payload.get("candidate_text")
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            continue
        key = candidate_text.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_aliases.append(candidate_text.strip())
    return unique_aliases


def _derive_item_status(action: str, resolution_scope: str) -> str:
    if action == ReviewAction.MARKED_NOISE.value:
        return (
            ReviewStatus.APPLIED.value
            if resolution_scope == ReviewResolutionScope.FUTURE_NOISE_SUPPRESSION.value
            else ReviewStatus.RESOLVED.value
        )
    if action in {ReviewAction.ACCEPTED.value, ReviewAction.REMAPPED.value}:
        return (
            ReviewStatus.APPLIED.value
            if resolution_scope == ReviewResolutionScope.FUTURE_OVERLAY.value
            else ReviewStatus.RESOLVED.value
        )
    if action == ReviewAction.REJECTED.value:
        return ReviewStatus.REJECTED.value
    if action == ReviewAction.KEPT_UNRESOLVED.value:
        return (
            ReviewStatus.OBSOLETE.value
            if resolution_scope == ReviewResolutionScope.CLOSE_ONLY.value
            else ReviewStatus.RESOLVED.value
        )
    return ReviewStatus.RESOLVED.value


def _default_review_context(settings: Settings | None) -> ReviewContext:
    if settings is None:
        return ReviewContext()
    from smap.ontology.runtime import load_runtime_ontology

    registry = load_runtime_ontology(settings).registry
    return ReviewContext(
        ontology_fingerprint=build_ontology_fingerprint(registry),
        overlay_fingerprint=build_overlay_fingerprint(registry),
        reviewed_overlay_fingerprint=build_reviewed_overlay_fingerprint(registry),
    )


def _review_context_for_mention(
    mention_id: str,
    review_contexts: Mapping[str, ReviewContext] | None,
    default_context: ReviewContext,
) -> ReviewContext:
    if review_contexts is None:
        return default_context
    context = review_contexts.get(mention_id)
    return context if context is not None else default_context


def _parse_knowledge_match_mode(value: str | None) -> KnowledgeMatchMode | None:
    if value is None:
        return None
    return KnowledgeMatchMode(value)


def _prepare_item_for_review(
    fact_payload: dict[str, Any],
    signature: SemanticSignature,
    context: ReviewContext,
    scope_key: ScopeKey,
    *,
    mention_id: str,
    source_uap_id: str,
    confidence: float,
    canonical_entity_id: str | None,
    group_fingerprint: str,
) -> ReviewItem:
    return ReviewItem(
        item_type=signature.item_type,
        problem_class=signature.problem_class.value,
        review_signature=group_fingerprint,
        semantic_fingerprint=signature.fingerprint(),
        scope_key_fingerprint=scope_key.fingerprint(),
        static_scope_fingerprint=scope_key.static_scope.fingerprint(),
        knowledge_state_fingerprint=scope_key.knowledge_state.fingerprint(),
        review_context_payload=context.model_dump(mode="json"),
        semantic_signature_payload=signature.model_dump(mode="json"),
        scope_key_payload=scope_key.model_dump(mode="json"),
        normalized_candidate_text=signature.normalized_candidate_text,
        entity_type_hint=signature.entity_type_hint,
        ambiguity_signature=signature.ambiguity_signature,
        mention_id=mention_id,
        source_uap_id=source_uap_id,
        confidence=confidence,
        status=ReviewStatus.PENDING.value,
        payload=fact_payload,
        canonical_entity_id=canonical_entity_id,
    )


def _prepare_overlay_provenance(
    *,
    base_provenance: dict[str, str | int | float | bool | None],
    context: ReviewContext,
    signature: SemanticSignature,
    applicability_policy: DecisionApplicabilityPolicy,
) -> dict[str, str | int | float | bool | None]:
    provenance = dict(base_provenance)
    provenance.update(applicability_to_provenance(applicability_policy))
    provenance["review_context_json"] = context.model_dump_json()
    provenance["semantic_signature_json"] = signature.model_dump_json()
    provenance["review_context_fingerprint"] = context.fingerprint()
    provenance["semantic_fingerprint"] = signature.fingerprint()
    return provenance


def _apply_overlay_updates(
    settings: Settings,
    *,
    aliases: list[str],
    remap_target: str | None,
    entity_type_hint: str | None,
    mark_as_noise: bool,
    provenance: dict[str, str | int | float | bool | None],
    notes: str | None,
    allow_authoritative_replacement: bool,
) -> None:
    overlay = _read_review_overlay(settings.review_overlay_path)
    overlay.metadata = overlay.metadata.model_copy(update={"version": utc_now().isoformat()})
    applicability_engine = ApplicabilityEngine()
    subsumption_engine = ContributionSubsumptionEngine(applicability_engine)
    incoming_policy = applicability_engine.policy_from_provenance(provenance)
    if incoming_policy is None:
        raise ValueError("Reviewed overlay update requires applicability provenance.")

    alias_keys = {normalize_alias(alias_text) for alias_text in aliases if normalize_alias(alias_text)}
    untouched_alias_contributions = [
        contribution
        for contribution in overlay.alias_contributions
        if normalize_alias(contribution.alias) not in alias_keys
    ]
    untouched_noise_terms = [
        noise_term
        for noise_term in overlay.noise_terms
        if normalize_alias(noise_term.term) not in alias_keys
    ]
    retained_alias_contributions: list[AliasContribution] = list(untouched_alias_contributions)
    retained_noise_terms: list[NoiseTerm] = list(untouched_noise_terms)

    for alias_key in sorted(alias_key for alias_key in alias_keys if alias_key is not None):
        related_aliases = [
            contribution
            for contribution in overlay.alias_contributions
            if normalize_alias(contribution.alias) == alias_key
        ]
        related_noise_terms = [
            noise_term
            for noise_term in overlay.noise_terms
            if normalize_alias(noise_term.term) == alias_key
        ]
        reconciliation = subsumption_engine.reconcile_contributions(
            existing_contributions=[*related_aliases, *related_noise_terms],
            incoming_policy=incoming_policy,
            incoming_target=remap_target,
            incoming_kind="alias" if remap_target is not None else "noise",
            allow_authoritative_replacement=allow_authoritative_replacement,
        )
        if reconciliation.action == ContributionAction.REJECT_CONFLICT:
            raise ValueError(f"Reviewed overlay contribution conflict for `{alias_key}`: {reconciliation.reason}")
        for contribution in reconciliation.retained_existing:
            if isinstance(contribution, AliasContribution):
                retained_alias_contributions.append(contribution)
            else:
                retained_noise_terms.append(contribution)

    overlay.alias_contributions = retained_alias_contributions
    overlay.noise_terms = retained_noise_terms
    existing_aliases = {
        (item.alias.casefold(), item.canonical_entity_id, json.dumps(item.provenance, sort_keys=True))
        for item in overlay.alias_contributions
    }
    existing_noise = {
        (item.term.casefold(), json.dumps(item.provenance, sort_keys=True))
        for item in overlay.noise_terms
    }

    if remap_target is not None:
        for alias_text in aliases:
            key = (alias_text.casefold(), remap_target, json.dumps(provenance, sort_keys=True))
            if key in existing_aliases:
                continue
            overlay.alias_contributions.append(
                AliasContribution(
                    canonical_entity_id=remap_target,
                    alias=alias_text,
                    entity_type=entity_type_hint,
                    source="review",
                    provenance=provenance,
                )
            )

    if mark_as_noise:
        for alias_text in aliases:
            noise_key = (alias_text.casefold(), json.dumps(provenance, sort_keys=True))
            if noise_key in existing_noise:
                continue
            overlay.noise_terms.append(
                NoiseTerm(
                    term=alias_text,
                    reason=notes or "review_marked_noise",
                    source="review",
                    provenance=provenance,
                )
            )

    _write_review_overlay(settings, overlay)


def _queue_entity_fact(
    session: Session,
    summary: ReviewQueueSummary,
    *,
    fact_payload: dict[str, Any],
    mention_id: str,
    source_uap_id: str,
    confidence: float,
    signature: SemanticSignature,
    context: ReviewContext,
    canonical_entity_id: str | None,
    policy_engine: ReviewPolicyEngine,
    applicability_engine: ApplicabilityEngine,
) -> None:
    summary.raw_reviewable_items += 1
    summary.entity_raw_reviewable_items += 1
    if canonical_entity_id is None:
        summary.unresolved_entity_candidates += 1

    decision = has_applicable_future_decision(
        session,
        signature=signature,
        context=context,
        applicability_engine=applicability_engine,
    )
    if decision is not None:
        group = get_review_group(session, decision.review_group_id)
        register_suppressed_occurrence(session, group)
        summary.suppressed_by_prior_decision += 1
        summary.entity_suppressed_items += 1
        return

    grouping_policy = policy_engine.grouping_policy_for(signature)
    scope_key = policy_engine.scope_key_for(
        context,
        grouping_policy.scope_level,
        grouping_policy=grouping_policy,
        signature=signature,
    )
    group_fingerprint = policy_engine.review_group_fingerprint(signature, scope_key)
    group = ensure_review_group(
        session,
        context=context,
        signature=signature,
        scope_key=scope_key,
        grouping_policy=grouping_policy,
        representative_payload=fact_payload,
        group_fingerprint=group_fingerprint,
        applicability_engine=applicability_engine,
    )
    review_item = _prepare_item_for_review(
        fact_payload,
        signature,
        context,
        scope_key,
        mention_id=mention_id,
        source_uap_id=source_uap_id,
        confidence=confidence,
        canonical_entity_id=canonical_entity_id,
        group_fingerprint=group_fingerprint,
    )
    session.add(review_item)
    session.flush()
    is_duplicate = attach_review_item_to_group(session, review_item, group)
    summary.created_items += 1
    summary.entity_created_items += 1
    if is_duplicate:
        summary.grouped_items += 1
        summary.entity_grouped_items += 1
        if canonical_entity_id is None:
            summary.repeat_unresolved_count += 1


def _queue_classification_fact(
    session: Session,
    summary: ReviewQueueSummary,
    *,
    fact_payload: dict[str, Any],
    mention_id: str,
    source_uap_id: str,
    confidence: float,
    signature: SemanticSignature,
    context: ReviewContext,
    policy_engine: ReviewPolicyEngine,
    applicability_engine: ApplicabilityEngine,
) -> None:
    summary.raw_reviewable_items += 1
    summary.classification_raw_reviewable_items += 1

    decision = has_applicable_future_decision(
        session,
        signature=signature,
        context=context,
        applicability_engine=applicability_engine,
    )
    if decision is not None:
        group = get_review_group(session, decision.review_group_id)
        register_suppressed_occurrence(session, group)
        summary.suppressed_by_prior_decision += 1
        summary.classification_suppressed_items += 1
        return

    grouping_policy = policy_engine.grouping_policy_for(signature)
    scope_key = policy_engine.scope_key_for(
        context,
        grouping_policy.scope_level,
        grouping_policy=grouping_policy,
        signature=signature,
    )
    group_fingerprint = policy_engine.review_group_fingerprint(signature, scope_key)
    group = ensure_review_group(
        session,
        context=context,
        signature=signature,
        scope_key=scope_key,
        grouping_policy=grouping_policy,
        representative_payload=fact_payload,
        group_fingerprint=group_fingerprint,
        applicability_engine=applicability_engine,
    )
    review_item = _prepare_item_for_review(
        fact_payload,
        signature,
        context,
        scope_key,
        mention_id=mention_id,
        source_uap_id=source_uap_id,
        confidence=confidence,
        canonical_entity_id=None,
        group_fingerprint=group_fingerprint,
    )
    session.add(review_item)
    session.flush()
    is_duplicate = attach_review_item_to_group(session, review_item, group)
    summary.created_items += 1
    summary.classification_created_items += 1
    if is_duplicate:
        summary.grouped_items += 1
        summary.classification_grouped_items += 1


def queue_review_items(
    session: Session,
    enrichment: EnrichmentBundle,
    *,
    settings: Settings | None = None,
    review_contexts: Mapping[str, ReviewContext] | None = None,
    policy_engine: ReviewPolicyEngine | None = None,
) -> ReviewQueueSummary:
    summary = ReviewQueueSummary()
    engine = policy_engine or ReviewPolicyEngine(settings)
    applicability_engine = ApplicabilityEngine(engine)
    default_context = _default_review_context(settings)

    for entity_fact in enrichment.entity_facts:
        if (
            entity_fact.canonical_entity_id is not None
            or entity_fact.concept_entity_id is not None
        ) and entity_fact.confidence >= 0.65:
            continue
        if entity_fact.unresolved_reason == "noise_term":
            continue
        signature = build_entity_signature(entity_fact)
        context = _review_context_for_mention(entity_fact.mention_id, review_contexts, default_context)
        _queue_entity_fact(
            session,
            summary,
            fact_payload=entity_fact.model_dump(mode="json"),
            mention_id=entity_fact.mention_id,
            source_uap_id=entity_fact.source_uap_id,
            confidence=entity_fact.confidence,
            signature=signature,
            context=context,
            canonical_entity_id=entity_fact.canonical_entity_id,
            policy_engine=engine,
            applicability_engine=applicability_engine,
        )

    reviewable_facts: list[SentimentFact | IntentFact | StanceFact] = [
        *enrichment.sentiment_facts,
        *enrichment.intent_facts,
        *enrichment.stance_facts,
    ]
    for fact in reviewable_facts:
        if fact.confidence >= 0.55:
            continue
        signature = build_classification_signature(fact)
        context = _review_context_for_mention(fact.mention_id, review_contexts, default_context)
        _queue_classification_fact(
            session,
            summary,
            fact_payload=fact.model_dump(mode="json"),
            mention_id=fact.mention_id,
            source_uap_id=fact.source_uap_id,
            confidence=fact.confidence,
            signature=signature,
            context=context,
            policy_engine=engine,
            applicability_engine=applicability_engine,
        )

    summary.active_groups = count_open_review_groups(session)
    return summary


def list_review_items(
    session: Session,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ReviewItem]:
    query: Select[tuple[ReviewItem]] = (
        select(ReviewItem)
        .options(selectinload(ReviewItem.review_group))
        .order_by(ReviewItem.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    if status:
        query = query.where(ReviewItem.status == status)
    return list(session.scalars(query))


def _update_item_resolution_state(
    review_item: ReviewItem,
    *,
    action: str,
    remap_target: str | None,
    resolution_scope: str,
) -> None:
    review_item.status = _derive_item_status(action, resolution_scope)
    review_item.resolved_at = utc_now()
    if remap_target is not None:
        review_item.canonical_entity_id = remap_target


def _item_semantics(
    review_item: ReviewItem,
    default_context: ReviewContext,
) -> tuple[ReviewContext, SemanticSignature, ScopeKey]:
    context = ReviewContext.model_validate(review_item.review_context_payload or default_context.model_dump(mode="json"))
    signature = (
        SemanticSignature.model_validate(review_item.semantic_signature_payload)
        if review_item.semantic_signature_payload is not None
        else build_signature_from_payload(review_item.item_type, review_item.payload)
    )
    if review_item.scope_key_payload is not None:
        scope_key = ScopeKey.model_validate(review_item.scope_key_payload)
    else:
        policy_engine = ReviewPolicyEngine()
        policy = policy_engine.grouping_policy_for(signature)
        scope_key = policy_engine.scope_key_for(
            context,
            policy.scope_level,
            grouping_policy=policy,
            signature=signature,
        )
    return context, signature, scope_key


def _group_semantics(group: ReviewGroup, default_context: ReviewContext) -> tuple[ReviewContext, SemanticSignature, ScopeKey]:
    context = get_group_context(group) or default_context
    group_signature = get_group_signature(group)
    if group_signature is not None:
        signature = group_signature
    else:
        inferred_item_type = "entity" if "candidate_text" in group.representative_payload else "classification"
        signature = build_signature_from_payload(inferred_item_type, group.representative_payload)
    policy_engine = ReviewPolicyEngine()
    grouping_policy = policy_engine.grouping_policy_for(signature)
    scope_key = get_group_scope_key(group) or policy_engine.scope_key_for(
        context,
        grouping_policy.scope_level,
        grouping_policy=grouping_policy,
        signature=signature,
    )
    return context, signature, scope_key


def _create_review_decision(
    *,
    review_item_id: int,
    action: str,
    reviewer: str,
    notes: str | None,
    remap_target: str | None,
    resolution_scope: str,
    context: ReviewContext,
    signature: SemanticSignature,
    applicability_policy: DecisionApplicabilityPolicy,
    supersedes_review_decision_id: int | None = None,
    terminated_by_review_decision_id: int | None = None,
    origin_group_decision_id: int | None = None,
    effect_applied_from_group_decision_id: int | None = None,
) -> ReviewDecision:
    return ReviewDecision(
        review_item_id=review_item_id,
        action=action,
        reviewer=reviewer,
        notes=notes,
        remap_target=remap_target,
        resolution_scope=resolution_scope,
        review_context_payload=context.model_dump(mode="json"),
        semantic_signature_payload=signature.model_dump(mode="json"),
        applicability_policy_payload=applicability_policy.model_dump(mode="json"),
        applicability_fingerprint=applicability_policy.fingerprint(),
        supersedes_review_decision_id=supersedes_review_decision_id,
        terminates_future_authority=applicability_policy.terminates_future_authority,
        terminated_by_review_decision_id=terminated_by_review_decision_id,
        origin_group_decision_id=origin_group_decision_id,
        effect_applied_from_group_decision_id=effect_applied_from_group_decision_id,
    )


def _create_group_decision(
    session: Session,
    *,
    group: ReviewGroup,
    action: str,
    reviewer: str,
    notes: str | None,
    remap_target: str | None,
    resolution_scope: str,
    context: ReviewContext,
    signature: SemanticSignature,
    applicability_policy: DecisionApplicabilityPolicy,
    applicability_engine: ApplicabilityEngine,
) -> ReviewGroupDecision:
    previous_effective = effective_future_group_decision(
        session,
        group.review_group_id,
        applicability_engine=applicability_engine,
    )
    group_decision = ReviewGroupDecision(
        review_group_id=group.review_group_id,
        action=action,
        reviewer=reviewer,
        notes=notes,
        remap_target=remap_target,
        resolution_scope=resolution_scope,
        review_context_payload=context.model_dump(mode="json"),
        semantic_signature_payload=signature.model_dump(mode="json"),
        applicability_policy_payload=applicability_policy.model_dump(mode="json"),
        applicability_fingerprint=applicability_policy.fingerprint(),
        terminates_future_authority=applicability_policy.terminates_future_authority,
        supersedes_review_group_decision_id=(
            previous_effective.review_group_decision_id
            if previous_effective is not None and applicability_policy.future_effect != FutureEffectType.NONE
            else None
        ),
    )
    session.add(group_decision)
    session.flush()
    if previous_effective is not None and applicability_policy.terminates_future_authority:
        previous_effective.terminated_by_review_group_decision_id = group_decision.review_group_decision_id
        session.add(previous_effective)
    return group_decision


def apply_review_decision(
    session: Session,
    review_item_id: int,
    decision: ReviewDecisionCreate,
    *,
    settings: Settings | None = None,
) -> ReviewItem:
    review_item = session.get(ReviewItem, review_item_id)
    if review_item is None:
        raise KeyError(review_item_id)

    default_context = _default_review_context(settings)
    context, signature, scope_key = _item_semantics(review_item, default_context)
    policy_engine = ReviewPolicyEngine(settings)
    applicability_engine = ApplicabilityEngine(policy_engine)
    applicability_policy = policy_engine.build_applicability_policy(
        signature,
        context,
        scope_key,
        resolution_scope=decision.resolution_scope,
        authority_level=AuthorityLevel.ITEM,
        knowledge_match_mode=_parse_knowledge_match_mode(decision.knowledge_match_mode),
        terminate_future_authority=decision.terminate_future_authority,
    )

    candidate_text = review_item.payload.get("candidate_text")
    alias_texts = [candidate_text] if isinstance(candidate_text, str) and candidate_text.strip() else []

    if decision.remap_target and settings is not None:
        validate_remap_target(settings, decision.remap_target)
    if settings is not None and decision.apply_alias_mapping and decision.remap_target and alias_texts:
        validate_alias_contribution(
            settings,
            aliases=alias_texts,
            remap_target=decision.remap_target,
            applicability_policy=applicability_policy,
            allow_replacement=applicability_policy.future_effect != FutureEffectType.NONE,
        )
    if settings is not None and decision.mark_as_noise and alias_texts:
        validate_noise_contribution(
            settings,
            aliases=alias_texts,
            applicability_policy=applicability_policy,
            allow_replacement=applicability_policy.future_effect != FutureEffectType.NONE,
        )

    previous_item_resolution = applicability_engine.item_authority_resolution(
        group_decisions=[],
        item_decisions=list(review_item.decisions),
    )
    previous_item_future = previous_item_resolution.authoritative_item_decision

    _update_item_resolution_state(
        review_item,
        action=decision.action,
        remap_target=decision.remap_target,
        resolution_scope=decision.resolution_scope,
    )
    persisted_decision = _create_review_decision(
        review_item_id=review_item_id,
        action=decision.action,
        reviewer=decision.reviewer,
        notes=decision.notes,
        remap_target=decision.remap_target,
        resolution_scope=decision.resolution_scope,
        context=context,
        signature=signature,
        applicability_policy=applicability_policy,
        supersedes_review_decision_id=(
            previous_item_future.review_decision_id
            if previous_item_future is not None and applicability_policy.future_effect != FutureEffectType.NONE
            else None
        ),
    )
    session.add(persisted_decision)
    session.add(review_item)
    session.flush()
    if previous_item_future is not None and applicability_policy.terminates_future_authority:
        previous_item_future.terminated_by_review_decision_id = persisted_decision.review_decision_id
        session.add(previous_item_future)

    if review_item.review_group_id is not None and applicability_policy.future_effect != FutureEffectType.NONE:
        group = get_review_group(session, review_item.review_group_id)
        group_context, group_signature, group_scope_key = _group_semantics(group, default_context)
        group_applicability = policy_engine.build_applicability_policy(
            group_signature,
            group_context,
            group_scope_key,
            resolution_scope=decision.resolution_scope,
            authority_level=AuthorityLevel.GROUP,
            knowledge_match_mode=_parse_knowledge_match_mode(decision.knowledge_match_mode),
            terminate_future_authority=decision.terminate_future_authority,
        )
        group_decision = _create_group_decision(
            session,
            group=group,
            action=decision.action,
            reviewer=decision.reviewer,
            notes=decision.notes,
            remap_target=decision.remap_target,
            resolution_scope=decision.resolution_scope,
            context=group_context,
            signature=group_signature,
            applicability_policy=group_applicability,
            applicability_engine=applicability_engine,
        )
        group_decision_history = list(group.decisions)
        if all(
            existing.review_group_decision_id != group_decision.review_group_decision_id
            for existing in group_decision_history
        ):
            group_decision_history.append(group_decision)
        group.status = _derive_item_status(decision.action, decision.resolution_scope)
        group.resolved_at = utc_now()
        group.active_item_count = 0
        for sibling in group.items:
            if sibling.review_item_id == review_item.review_item_id or sibling.status not in OPEN_REVIEW_STATUSES:
                continue
            sibling_context, sibling_signature, sibling_scope_key = _item_semantics(sibling, default_context)
            sibling_applicability = policy_engine.build_applicability_policy(
                sibling_signature,
                sibling_context,
                sibling_scope_key,
                resolution_scope=decision.resolution_scope,
                authority_level=AuthorityLevel.ITEM,
                knowledge_match_mode=_parse_knowledge_match_mode(decision.knowledge_match_mode),
                terminate_future_authority=decision.terminate_future_authority,
            )
            previous_sibling_resolution = applicability_engine.item_authority_resolution(
                group_decisions=group_decision_history,
                item_decisions=list(sibling.decisions),
            )
            previous_sibling_future = previous_sibling_resolution.authoritative_item_decision
            _update_item_resolution_state(
                sibling,
                action=decision.action,
                remap_target=decision.remap_target,
                resolution_scope=decision.resolution_scope,
            )
            propagated_decision = _create_review_decision(
                review_item_id=sibling.review_item_id,
                action=decision.action,
                reviewer=decision.reviewer,
                notes=decision.notes,
                remap_target=decision.remap_target,
                resolution_scope=decision.resolution_scope,
                context=sibling_context,
                signature=sibling_signature,
                applicability_policy=sibling_applicability,
                supersedes_review_decision_id=(
                    previous_sibling_future.review_decision_id
                    if previous_sibling_future is not None and sibling_applicability.future_effect != FutureEffectType.NONE
                    else None
                ),
                origin_group_decision_id=group_decision.review_group_decision_id,
                effect_applied_from_group_decision_id=group_decision.review_group_decision_id,
            )
            session.add(propagated_decision)
            session.add(sibling)
            session.flush()
            if previous_sibling_future is not None and sibling_applicability.terminates_future_authority:
                previous_sibling_future.terminated_by_review_decision_id = propagated_decision.review_decision_id
                session.add(previous_sibling_future)
        session.add(group)

    if settings is not None and (decision.apply_alias_mapping or decision.mark_as_noise) and alias_texts:
        provenance = _prepare_overlay_provenance(
            base_provenance={
                "review_item_id": review_item.review_item_id,
                "review_decision_id": persisted_decision.review_decision_id,
                "reviewer": persisted_decision.reviewer,
                "action": persisted_decision.action,
                "notes": persisted_decision.notes,
            },
            context=context,
            signature=signature,
            applicability_policy=applicability_policy,
        )
        _apply_overlay_updates(
            settings,
            aliases=alias_texts,
            remap_target=decision.remap_target if decision.apply_alias_mapping else None,
            entity_type_hint=review_item.entity_type_hint,
            mark_as_noise=decision.mark_as_noise,
            provenance=provenance,
            notes=decision.notes,
            allow_authoritative_replacement=applicability_policy.future_effect != FutureEffectType.NONE,
        )
    return review_item


def apply_review_group_decision(
    session: Session,
    review_group_id: int,
    decision: ReviewGroupDecisionCreate,
    *,
    settings: Settings | None = None,
) -> ReviewGroup:
    group = get_review_group(session, review_group_id)
    default_context = _default_review_context(settings)
    context, signature, scope_key = _group_semantics(group, default_context)
    policy_engine = ReviewPolicyEngine(settings)
    applicability_engine = ApplicabilityEngine(policy_engine)
    applicability_policy = policy_engine.build_applicability_policy(
        signature,
        context,
        scope_key,
        resolution_scope=decision.resolution_scope,
        authority_level=AuthorityLevel.GROUP,
        knowledge_match_mode=_parse_knowledge_match_mode(decision.knowledge_match_mode),
        terminate_future_authority=decision.terminate_future_authority,
    )
    alias_texts = list(iter_group_candidate_texts(group))

    if decision.remap_target and settings is not None:
        validate_remap_target(settings, decision.remap_target)
    if settings is not None and decision.apply_alias_mapping and decision.remap_target and alias_texts:
        validate_alias_contribution(
            settings,
            aliases=alias_texts,
            remap_target=decision.remap_target,
            applicability_policy=applicability_policy,
            allow_replacement=applicability_policy.future_effect != FutureEffectType.NONE,
        )
    if settings is not None and decision.mark_as_noise and alias_texts:
        validate_noise_contribution(
            settings,
            aliases=alias_texts,
            applicability_policy=applicability_policy,
            allow_replacement=applicability_policy.future_effect != FutureEffectType.NONE,
        )

    group_decision = _create_group_decision(
        session,
        group=group,
        action=decision.action,
        reviewer=decision.reviewer,
        notes=decision.notes,
        remap_target=decision.remap_target,
        resolution_scope=decision.resolution_scope,
        context=context,
        signature=signature,
        applicability_policy=applicability_policy,
        applicability_engine=applicability_engine,
    )
    group_decision_history = list(group.decisions)
    if all(
        existing.review_group_decision_id != group_decision.review_group_decision_id
        for existing in group_decision_history
    ):
        group_decision_history.append(group_decision)

    item_status = _derive_item_status(decision.action, decision.resolution_scope)
    group.status = item_status
    group.resolved_at = utc_now()
    group.active_item_count = 0
    session.add(group)

    for item in group.items:
        if item.status not in OPEN_REVIEW_STATUSES:
            continue
        item_context, item_signature, item_scope_key = _item_semantics(item, default_context)
        item_applicability = policy_engine.build_applicability_policy(
            item_signature,
            item_context,
            item_scope_key,
            resolution_scope=decision.resolution_scope,
            authority_level=AuthorityLevel.ITEM,
            knowledge_match_mode=_parse_knowledge_match_mode(decision.knowledge_match_mode),
            terminate_future_authority=decision.terminate_future_authority,
        )
        previous_item_resolution = applicability_engine.item_authority_resolution(
            group_decisions=group_decision_history,
            item_decisions=list(item.decisions),
        )
        previous_item_future = previous_item_resolution.authoritative_item_decision
        _update_item_resolution_state(
            item,
            action=decision.action,
            remap_target=decision.remap_target,
            resolution_scope=decision.resolution_scope,
        )
        propagated_decision = _create_review_decision(
            review_item_id=item.review_item_id,
            action=decision.action,
            reviewer=decision.reviewer,
            notes=decision.notes,
            remap_target=decision.remap_target,
            resolution_scope=decision.resolution_scope,
            context=item_context,
            signature=item_signature,
            applicability_policy=item_applicability,
            supersedes_review_decision_id=(
                previous_item_future.review_decision_id
                if previous_item_future is not None and item_applicability.future_effect != FutureEffectType.NONE
                else None
            ),
            origin_group_decision_id=group_decision.review_group_decision_id,
            effect_applied_from_group_decision_id=group_decision.review_group_decision_id,
        )
        session.add(propagated_decision)
        session.add(item)
        session.flush()
        if previous_item_future is not None and item_applicability.terminates_future_authority:
            previous_item_future.terminated_by_review_decision_id = propagated_decision.review_decision_id
            session.add(previous_item_future)

    if settings is not None and (decision.apply_alias_mapping or decision.mark_as_noise) and alias_texts:
        provenance = _prepare_overlay_provenance(
            base_provenance={
                "review_group_id": group.review_group_id,
                "review_group_decision_id": group_decision.review_group_decision_id,
                "reviewer": group_decision.reviewer,
                "action": group_decision.action,
                "notes": group_decision.notes,
            },
            context=context,
            signature=signature,
            applicability_policy=applicability_policy,
        )
        _apply_overlay_updates(
            settings,
            aliases=alias_texts,
            remap_target=decision.remap_target if decision.apply_alias_mapping else None,
            entity_type_hint=group.entity_type_hint,
            mark_as_noise=decision.mark_as_noise,
            provenance=provenance,
            notes=decision.notes,
            allow_authoritative_replacement=applicability_policy.future_effect != FutureEffectType.NONE,
        )

    return group


def export_review_items(session: Session, path: Path, status: str = "pending") -> int:
    items = list_review_items(session, status=status, limit=100000, offset=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(
                json.dumps(
                    {
                        "review_item_id": item.review_item_id,
                        "item_type": item.item_type,
                        "problem_class": item.problem_class,
                        "review_signature": item.review_signature,
                        "semantic_fingerprint": item.semantic_fingerprint,
                        "scope_key_fingerprint": item.scope_key_fingerprint,
                        "static_scope_fingerprint": item.static_scope_fingerprint,
                        "knowledge_state_fingerprint": item.knowledge_state_fingerprint,
                        "review_context_payload": item.review_context_payload,
                        "semantic_signature_payload": item.semantic_signature_payload,
                        "scope_key_payload": item.scope_key_payload,
                        "normalized_candidate_text": item.normalized_candidate_text,
                        "entity_type_hint": item.entity_type_hint,
                        "ambiguity_signature": item.ambiguity_signature,
                        "mention_id": item.mention_id,
                        "source_uap_id": item.source_uap_id,
                        "confidence": item.confidence,
                        "status": item.status,
                        "payload": item.payload,
                        "canonical_entity_id": item.canonical_entity_id,
                        "review_group_id": item.review_group_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(items)


def export_review_groups(session: Session, path: Path, status: str | None = None) -> int:
    groups = list_review_groups(session, status=status, limit=100000, offset=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(
                json.dumps(
                    {
                        "review_group_id": group.review_group_id,
                        "problem_class": group.problem_class,
                        "review_signature": group.review_signature,
                        "semantic_fingerprint": group.semantic_fingerprint,
                        "scope_key_fingerprint": group.scope_key_fingerprint,
                        "static_scope_fingerprint": group.static_scope_fingerprint,
                        "knowledge_state_fingerprint": group.knowledge_state_fingerprint,
                        "review_context_payload": group.review_context_payload,
                        "semantic_signature_payload": group.semantic_signature_payload,
                        "scope_key_payload": group.scope_key_payload,
                        "grouping_policy_payload": group.grouping_policy_payload,
                        "normalized_candidate_text": group.normalized_candidate_text,
                        "entity_type_hint": group.entity_type_hint,
                        "ambiguity_signature": group.ambiguity_signature,
                        "candidate_canonical_ids": group.candidate_canonical_ids,
                        "occurrence_count": group.occurrence_count,
                        "active_item_count": group.active_item_count,
                        "status": group.status,
                        "assignee": group.assignee,
                        "representative_payload": group.representative_payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(groups)


def import_review_decisions(session: Session, path: Path, *, settings: Settings | None = None) -> int:
    applied = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            apply_review_decision(
                session,
                review_item_id=int(payload["review_item_id"]),
                decision=ReviewDecisionCreate(
                    action=payload["action"],
                    reviewer=payload.get("reviewer", "import"),
                    notes=payload.get("notes"),
                    remap_target=payload.get("remap_target"),
                    apply_alias_mapping=bool(payload.get("apply_alias_mapping", False)),
                    mark_as_noise=bool(payload.get("mark_as_noise", False)),
                    resolution_scope=payload.get("resolution_scope", ReviewResolutionScope.ITEM_ONLY.value),
                ),
                settings=settings,
            )
            applied += 1
    return applied
