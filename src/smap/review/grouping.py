from __future__ import annotations
from collections.abc import Iterable
from typing import cast
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, selectinload
from smap.core.types import utc_now
from smap.review.applicability_engine import ApplicabilityEngine
from smap.review.context import ReviewContext, ScopeKey, SemanticSignature
from smap.review.models import ReviewGroup, ReviewGroupDecision, ReviewItem
from smap.review.policy import DecisionApplicabilityPolicy, ReviewGroupingPolicy, ReviewPolicyEngine
from smap.review.signatures import build_signature_from_payload
from smap.review.types import FutureEffectType, ReviewStatus
OPEN_REVIEW_STATUSES = {ReviewStatus.PENDING.value, ReviewStatus.GROUPED.value, ReviewStatus.ASSIGNED.value, ReviewStatus.IN_REVIEW.value}

def list_review_groups(session, *, status=None, problem_class=None, limit=50, offset=0):
    query = select(ReviewGroup).options(selectinload(ReviewGroup.items), selectinload(ReviewGroup.decisions)).order_by(ReviewGroup.updated_at.desc()).offset(offset).limit(limit)
    if status:
        query = query.where(ReviewGroup.status == status)
    if problem_class:
        query = query.where(ReviewGroup.problem_class == problem_class)
    return list(session.scalars(query))

def get_review_group(session, review_group_id):
    group = session.scalar(select(ReviewGroup).options(selectinload(ReviewGroup.items), selectinload(ReviewGroup.decisions)).where(ReviewGroup.review_group_id == review_group_id))
    if group is None:
        raise KeyError(review_group_id)
    return group

def count_open_review_groups(session):
    result = session.scalar(select(func.count()).select_from(ReviewGroup).where(ReviewGroup.status.in_(OPEN_REVIEW_STATUSES)))
    return int(result or 0)

def list_group_decisions(session, review_group_id):
    return list(session.scalars(select(ReviewGroupDecision).where(ReviewGroupDecision.review_group_id == review_group_id).order_by(ReviewGroupDecision.created_at.asc(), ReviewGroupDecision.review_group_decision_id.asc())))

def latest_group_decision(session, review_group_id):
    return session.scalar(select(ReviewGroupDecision).where(ReviewGroupDecision.review_group_id == review_group_id).order_by(ReviewGroupDecision.created_at.desc(), ReviewGroupDecision.review_group_decision_id.desc()).limit(1))

def effective_future_group_decision(session, review_group_id, *, applicability_engine=None):
    engine = applicability_engine or ApplicabilityEngine()
    return cast(ReviewGroupDecision | None, engine.effective_future_decision(list_group_decisions(session, review_group_id)))

def find_group_by_fingerprint(session, group_fingerprint):
    return session.scalar(select(ReviewGroup).options(selectinload(ReviewGroup.decisions)).where(ReviewGroup.review_signature == group_fingerprint))

def find_candidate_groups(session, *, semantic_fingerprint, problem_class):
    query = select(ReviewGroup).options(selectinload(ReviewGroup.decisions)).where(ReviewGroup.problem_class == problem_class).where(ReviewGroup.semantic_fingerprint == semantic_fingerprint)
    return list(session.scalars(query))

def get_group_context(group):
    if group.review_context_payload is None:
        return None
    return ReviewContext.model_validate(group.review_context_payload)

def get_group_signature(group):
    if group.semantic_signature_payload is None:
        return None
    return SemanticSignature.model_validate(group.semantic_signature_payload)

def get_group_scope_key(group):
    if group.scope_key_payload is None:
        return None
    return ScopeKey.model_validate(group.scope_key_payload)

def get_grouping_policy(group):
    if group.grouping_policy_payload is None:
        return None
    return ReviewGroupingPolicy.model_validate(group.grouping_policy_payload)

def get_applicability_policy(decision):
    if decision.applicability_policy_payload is None:
        return None
    return DecisionApplicabilityPolicy.model_validate(decision.applicability_policy_payload)

def has_applicable_future_decision(session, *, signature, context, applicability_engine=None):
    engine = applicability_engine or ApplicabilityEngine()
    candidate_groups = find_candidate_groups(session, semantic_fingerprint=signature.fingerprint(), problem_class=signature.problem_class.value)
    for group in candidate_groups:
        decision = effective_future_group_decision(session, group.review_group_id, applicability_engine=engine)
        if decision is None:
            continue
        applicability = get_applicability_policy(decision)
        if applicability is None:
            continue
        if engine.decision_applies(applicability, signature, context):
            return decision
    return None

def ensure_review_group(session, *, context, signature, scope_key, grouping_policy, representative_payload, group_fingerprint, applicability_engine=None):
    group = find_group_by_fingerprint(session, group_fingerprint)
    if group is None:
        group = ReviewGroup(problem_class=signature.problem_class.value, review_signature=group_fingerprint, semantic_fingerprint=signature.fingerprint(), scope_key_fingerprint=scope_key.fingerprint(), static_scope_fingerprint=scope_key.static_scope.fingerprint(), knowledge_state_fingerprint=scope_key.knowledge_state.fingerprint(), review_context_payload=context.model_dump(mode='json'), semantic_signature_payload=signature.model_dump(mode='json'), scope_key_payload=scope_key.model_dump(mode='json'), grouping_policy_payload=grouping_policy.model_dump(mode='json'), normalized_candidate_text=signature.normalized_candidate_text, entity_type_hint=signature.entity_type_hint, ambiguity_signature=signature.ambiguity_signature, candidate_canonical_ids=signature.candidate_canonical_ids, representative_payload=representative_payload, occurrence_count=0, active_item_count=0, status=ReviewStatus.PENDING.value, last_seen_at=utc_now())
        session.add(group)
        session.flush()
        return group
    if group.status not in OPEN_REVIEW_STATUSES:
        decision = effective_future_group_decision(session, group.review_group_id, applicability_engine=applicability_engine)
        applicability = get_applicability_policy(decision) if decision else None
        if applicability is None or applicability.future_effect == FutureEffectType.NONE:
            group.status = ReviewStatus.PENDING.value
            group.assignee = None
            group.assigned_at = None
            group.resolved_at = None
            group.active_item_count = 0
    group.last_seen_at = utc_now()
    session.add(group)
    return group

def attach_review_item_to_group(session, review_item, group):
    is_duplicate = group.occurrence_count > 0
    group.occurrence_count += 1
    group.active_item_count += 1
    group.last_seen_at = utc_now()
    review_item.review_group_id = group.review_group_id
    review_item.status = ReviewStatus.GROUPED.value if is_duplicate else ReviewStatus.PENDING.value
    session.add(group)
    session.add(review_item)
    return is_duplicate

def register_suppressed_occurrence(session, group):
    group.occurrence_count += 1
    group.last_seen_at = utc_now()
    session.add(group)

def assign_review_group(session, review_group_id, assignee):
    group = get_review_group(session, review_group_id)
    timestamp = utc_now()
    group.assignee = assignee
    group.assigned_at = timestamp
    group.status = ReviewStatus.ASSIGNED.value
    for item in group.items:
        if item.status in OPEN_REVIEW_STATUSES:
            item.assignee = assignee
            item.assigned_at = timestamp
            item.status = ReviewStatus.ASSIGNED.value
            session.add(item)
    session.add(group)
    return group

def rebuild_review_groups(session, policy_engine=None):
    active_items = list(session.scalars(select(ReviewItem).where(ReviewItem.status.in_(OPEN_REVIEW_STATUSES)).order_by(ReviewItem.created_at.asc())))
    for group in list(session.scalars(select(ReviewGroup))):
        group.active_item_count = 0
        if group.status in OPEN_REVIEW_STATUSES:
            group.occurrence_count = 0
        session.add(group)
    engine = policy_engine or ReviewPolicyEngine()
    applicability_engine = ApplicabilityEngine(engine)
    rebuilt = 0
    for item in active_items:
        context_payload = item.review_context_payload or {}
        signature_payload = item.semantic_signature_payload
        scope_key_payload = item.scope_key_payload
        if not signature_payload:
            signature = build_signature_from_payload(item.item_type, item.payload)
        else:
            signature = SemanticSignature.model_validate(signature_payload)
        context = ReviewContext.model_validate(context_payload)
        grouping_policy = engine.grouping_policy_for(signature)
        scope_key = ScopeKey.model_validate(scope_key_payload) if scope_key_payload else engine.scope_key_for(context, grouping_policy.scope_level, grouping_policy=grouping_policy, signature=signature)
        item.static_scope_fingerprint = scope_key.static_scope.fingerprint()
        item.knowledge_state_fingerprint = scope_key.knowledge_state.fingerprint()
        item.scope_key_fingerprint = scope_key.fingerprint()
        item.semantic_fingerprint = signature.fingerprint()
        item.semantic_signature_payload = signature.model_dump(mode='json')
        item.scope_key_payload = scope_key.model_dump(mode='json')
        group = ensure_review_group(session, context=context, signature=signature, scope_key=scope_key, grouping_policy=grouping_policy, representative_payload=item.payload, group_fingerprint=engine.review_group_fingerprint(signature, scope_key), applicability_engine=applicability_engine)
        attach_review_item_to_group(session, item, group)
        rebuilt += 1
    return rebuilt

def iter_group_candidate_texts(group):
    seen = set()
    for item in group.items:
        candidate_text = item.payload.get('candidate_text')
        if isinstance(candidate_text, str) and candidate_text.strip() and (candidate_text not in seen):
            seen.add(candidate_text)
            yield candidate_text
