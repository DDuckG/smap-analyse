from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from smap.review.context import ReviewContext, ScopeKey, SemanticSignature
from smap.review.policy import DecisionApplicabilityPolicy, ReviewGroupingPolicy
from smap.review.types import ReviewResolutionScope


class ReviewItemRead(BaseModel):
    review_item_id: int
    item_type: str
    problem_class: str
    review_signature: str
    semantic_fingerprint: str | None = None
    scope_key_fingerprint: str | None = None
    review_context: ReviewContext | None = None
    semantic_signature: SemanticSignature | None = None
    scope_key: ScopeKey | None = None
    normalized_candidate_text: str | None = None
    entity_type_hint: str | None = None
    ambiguity_signature: str | None = None
    mention_id: str
    source_uap_id: str
    confidence: float
    status: str
    payload: dict[str, Any]
    canonical_entity_id: str | None = None
    review_group_id: int | None = None
    assignee: str | None = None
    assigned_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None


class ReviewGroupRead(BaseModel):
    review_group_id: int
    problem_class: str
    review_signature: str
    semantic_fingerprint: str | None = None
    scope_key_fingerprint: str | None = None
    review_context: ReviewContext | None = None
    semantic_signature: SemanticSignature | None = None
    scope_key: ScopeKey | None = None
    grouping_policy: ReviewGroupingPolicy | None = None
    normalized_candidate_text: str | None = None
    entity_type_hint: str | None = None
    ambiguity_signature: str | None = None
    candidate_canonical_ids: list[str] = Field(default_factory=list)
    occurrence_count: int
    active_item_count: int
    status: str
    assignee: str | None = None
    assigned_at: datetime | None = None
    resolved_at: datetime | None = None
    representative_payload: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_seen_at: datetime


class ReviewDecisionCreate(BaseModel):
    action: str
    reviewer: str = "api"
    notes: str | None = None
    remap_target: str | None = None
    apply_alias_mapping: bool = False
    mark_as_noise: bool = False
    resolution_scope: str = ReviewResolutionScope.ITEM_ONLY.value
    knowledge_match_mode: str | None = None
    terminate_future_authority: bool | None = None


class ReviewGroupDecisionCreate(BaseModel):
    action: str
    reviewer: str = "api"
    notes: str | None = None
    remap_target: str | None = None
    apply_alias_mapping: bool = False
    mark_as_noise: bool = False
    resolution_scope: str = ReviewResolutionScope.GROUP.value
    knowledge_match_mode: str | None = None
    terminate_future_authority: bool | None = None


class ReviewDecisionRead(BaseModel):
    review_context: ReviewContext | None = None
    semantic_signature: SemanticSignature | None = None
    applicability_policy: DecisionApplicabilityPolicy | None = None
