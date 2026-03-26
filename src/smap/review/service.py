from smap.review.applicability_engine import ApplicabilityEngine
from smap.review.context import ReviewContext, build_ontology_fingerprint, build_overlay_fingerprint, build_reviewed_overlay_fingerprint
from smap.review.grouping import attach_review_item_to_group, count_open_review_groups, ensure_review_group, get_review_group, has_applicable_future_decision, register_suppressed_occurrence
from smap.review.metrics import ReviewQueueSummary
from smap.review.models import OntologyRegistryVersion, ReviewItem
from smap.review.policy import ReviewPolicyEngine
from smap.review.signatures import build_classification_signature, build_entity_signature
from smap.review.types import ReviewStatus

def seed_ontology_version(session, registry):
    existing = session.get(OntologyRegistryVersion, registry.metadata.version)
    if existing is None:
        session.add(OntologyRegistryVersion(version_id=registry.metadata.version, name=registry.metadata.name, description=registry.metadata.description))

def _default_review_context(settings):
    if settings is None:
        return ReviewContext()
    from smap.ontology.runtime import load_runtime_ontology
    registry = load_runtime_ontology(settings).registry
    return ReviewContext(ontology_fingerprint=build_ontology_fingerprint(registry), overlay_fingerprint=build_overlay_fingerprint(registry), reviewed_overlay_fingerprint=build_reviewed_overlay_fingerprint(registry))

def _review_context_for_mention(mention_id, review_contexts, default_context):
    if review_contexts is None:
        return default_context
    context = review_contexts.get(mention_id)
    return context if context is not None else default_context

def _prepare_item_for_review(fact_payload, signature, context, scope_key, *, mention_id, source_uap_id, confidence, canonical_entity_id, group_fingerprint):
    return ReviewItem(item_type=signature.item_type, problem_class=signature.problem_class.value, review_signature=group_fingerprint, semantic_fingerprint=signature.fingerprint(), scope_key_fingerprint=scope_key.fingerprint(), static_scope_fingerprint=scope_key.static_scope.fingerprint(), knowledge_state_fingerprint=scope_key.knowledge_state.fingerprint(), review_context_payload=context.model_dump(mode='json'), semantic_signature_payload=signature.model_dump(mode='json'), scope_key_payload=scope_key.model_dump(mode='json'), normalized_candidate_text=signature.normalized_candidate_text, entity_type_hint=signature.entity_type_hint, ambiguity_signature=signature.ambiguity_signature, mention_id=mention_id, source_uap_id=source_uap_id, confidence=confidence, status=ReviewStatus.PENDING.value, payload=fact_payload, canonical_entity_id=canonical_entity_id)

def _queue_entity_fact(session, summary, *, fact_payload, mention_id, source_uap_id, confidence, signature, context, canonical_entity_id, policy_engine, applicability_engine):
    summary.raw_reviewable_items += 1
    summary.entity_raw_reviewable_items += 1
    if canonical_entity_id is None:
        summary.unresolved_entity_candidates += 1
    decision = has_applicable_future_decision(session, signature=signature, context=context, applicability_engine=applicability_engine)
    if decision is not None:
        group = get_review_group(session, decision.review_group_id)
        register_suppressed_occurrence(session, group)
        summary.suppressed_by_prior_decision += 1
        summary.entity_suppressed_items += 1
        return
    grouping_policy = policy_engine.grouping_policy_for(signature)
    scope_key = policy_engine.scope_key_for(context, grouping_policy.scope_level, grouping_policy=grouping_policy, signature=signature)
    group_fingerprint = policy_engine.review_group_fingerprint(signature, scope_key)
    group = ensure_review_group(session, context=context, signature=signature, scope_key=scope_key, grouping_policy=grouping_policy, representative_payload=fact_payload, group_fingerprint=group_fingerprint, applicability_engine=applicability_engine)
    review_item = _prepare_item_for_review(fact_payload, signature, context, scope_key, mention_id=mention_id, source_uap_id=source_uap_id, confidence=confidence, canonical_entity_id=canonical_entity_id, group_fingerprint=group_fingerprint)
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

def _queue_classification_fact(session, summary, *, fact_payload, mention_id, source_uap_id, confidence, signature, context, policy_engine, applicability_engine):
    summary.raw_reviewable_items += 1
    summary.classification_raw_reviewable_items += 1
    decision = has_applicable_future_decision(session, signature=signature, context=context, applicability_engine=applicability_engine)
    if decision is not None:
        group = get_review_group(session, decision.review_group_id)
        register_suppressed_occurrence(session, group)
        summary.suppressed_by_prior_decision += 1
        summary.classification_suppressed_items += 1
        return
    grouping_policy = policy_engine.grouping_policy_for(signature)
    scope_key = policy_engine.scope_key_for(context, grouping_policy.scope_level, grouping_policy=grouping_policy, signature=signature)
    group_fingerprint = policy_engine.review_group_fingerprint(signature, scope_key)
    group = ensure_review_group(session, context=context, signature=signature, scope_key=scope_key, grouping_policy=grouping_policy, representative_payload=fact_payload, group_fingerprint=group_fingerprint, applicability_engine=applicability_engine)
    review_item = _prepare_item_for_review(fact_payload, signature, context, scope_key, mention_id=mention_id, source_uap_id=source_uap_id, confidence=confidence, canonical_entity_id=None, group_fingerprint=group_fingerprint)
    session.add(review_item)
    session.flush()
    is_duplicate = attach_review_item_to_group(session, review_item, group)
    summary.created_items += 1
    summary.classification_created_items += 1
    if is_duplicate:
        summary.grouped_items += 1
        summary.classification_grouped_items += 1

def queue_review_items(session, enrichment, *, settings=None, review_contexts=None, policy_engine=None):
    summary = ReviewQueueSummary()
    engine = policy_engine or ReviewPolicyEngine(settings)
    applicability_engine = ApplicabilityEngine(engine)
    default_context = _default_review_context(settings)
    for entity_fact in enrichment.entity_facts:
        if (entity_fact.canonical_entity_id is not None or entity_fact.concept_entity_id is not None) and entity_fact.confidence >= 0.65:
            continue
        if entity_fact.unresolved_reason == 'noise_term':
            continue
        signature = build_entity_signature(entity_fact)
        context = _review_context_for_mention(entity_fact.mention_id, review_contexts, default_context)
        _queue_entity_fact(session, summary, fact_payload=entity_fact.model_dump(mode='json'), mention_id=entity_fact.mention_id, source_uap_id=entity_fact.source_uap_id, confidence=entity_fact.confidence, signature=signature, context=context, canonical_entity_id=entity_fact.canonical_entity_id, policy_engine=engine, applicability_engine=applicability_engine)
    reviewable_facts = [*enrichment.sentiment_facts, *enrichment.intent_facts, *enrichment.stance_facts]
    for fact in reviewable_facts:
        if fact.confidence >= 0.55:
            continue
        signature = build_classification_signature(fact)
        context = _review_context_for_mention(fact.mention_id, review_contexts, default_context)
        _queue_classification_fact(session, summary, fact_payload=fact.model_dump(mode='json'), mention_id=fact.mention_id, source_uap_id=fact.source_uap_id, confidence=fact.confidence, signature=signature, context=context, policy_engine=engine, applicability_engine=applicability_engine)
    summary.active_groups = count_open_review_groups(session)
    return summary
