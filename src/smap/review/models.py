from __future__ import annotations
from datetime import datetime
from typing import Any
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from smap.core.types import utc_now
from smap.review.types import ReviewResolutionScope, ReviewStatus

class Base(DeclarativeBase):
    pass

class OntologyRegistryVersion(Base):
    __tablename__ = 'ontology_registry_versions'
    version_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

class ReviewGroup(Base):
    __tablename__ = 'review_groups'
    review_group_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    problem_class: Mapped[str] = mapped_column(String(80), index=True)
    review_signature: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    semantic_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    scope_key_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    static_scope_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    knowledge_state_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    review_context_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    semantic_signature_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    scope_key_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    grouping_policy_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    normalized_candidate_text: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    entity_type_hint: Mapped[str | None] = mapped_column(String(80), nullable=True)
    ambiguity_signature: Mapped[str | None] = mapped_column(String(255), nullable=True)
    candidate_canonical_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    representative_payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=0)
    active_item_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(50), default=ReviewStatus.PENDING.value, index=True)
    assignee: Mapped[str | None] = mapped_column(String(120), nullable=True)
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    items: Mapped[list[ReviewItem]] = relationship(back_populates='review_group')
    decisions: Mapped[list[ReviewGroupDecision]] = relationship(back_populates='review_group')

class ReviewItem(Base):
    __tablename__ = 'review_items'
    review_item_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    item_type: Mapped[str] = mapped_column(String(50))
    problem_class: Mapped[str] = mapped_column(String(80), index=True)
    review_signature: Mapped[str] = mapped_column(String(255), index=True)
    semantic_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    scope_key_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    static_scope_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    knowledge_state_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    review_context_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    semantic_signature_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    scope_key_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    normalized_candidate_text: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    entity_type_hint: Mapped[str | None] = mapped_column(String(80), nullable=True)
    ambiguity_signature: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mention_id: Mapped[str] = mapped_column(String(200), index=True)
    source_uap_id: Mapped[str] = mapped_column(String(200), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(50), default=ReviewStatus.PENDING.value, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    canonical_entity_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    review_group_id: Mapped[int | None] = mapped_column(ForeignKey('review_groups.review_group_id'), nullable=True, index=True)
    assignee: Mapped[str | None] = mapped_column(String(120), nullable=True)
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    decisions: Mapped[list[ReviewDecision]] = relationship(back_populates='review_item')
    review_group: Mapped[ReviewGroup | None] = relationship(back_populates='items')

class ReviewDecision(Base):
    __tablename__ = 'review_decisions'
    review_decision_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    review_item_id: Mapped[int] = mapped_column(ForeignKey('review_items.review_item_id'), index=True)
    action: Mapped[str] = mapped_column(String(50))
    reviewer: Mapped[str] = mapped_column(String(100), default='system')
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    remap_target: Mapped[str | None] = mapped_column(String(200), nullable=True)
    resolution_scope: Mapped[str] = mapped_column(String(80), default=ReviewResolutionScope.ITEM_ONLY.value)
    review_context_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    semantic_signature_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    applicability_policy_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    applicability_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    supersedes_review_decision_id: Mapped[int | None] = mapped_column(ForeignKey('review_decisions.review_decision_id'), nullable=True, index=True)
    terminates_future_authority: Mapped[bool] = mapped_column(default=False)
    terminated_by_review_decision_id: Mapped[int | None] = mapped_column(ForeignKey('review_decisions.review_decision_id'), nullable=True, index=True)
    origin_group_decision_id: Mapped[int | None] = mapped_column(ForeignKey('review_group_decisions.review_group_decision_id'), nullable=True, index=True)
    effect_applied_from_group_decision_id: Mapped[int | None] = mapped_column(ForeignKey('review_group_decisions.review_group_decision_id'), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    review_item: Mapped[ReviewItem] = relationship(back_populates='decisions')

class ReviewGroupDecision(Base):
    __tablename__ = 'review_group_decisions'
    review_group_decision_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    review_group_id: Mapped[int] = mapped_column(ForeignKey('review_groups.review_group_id'), index=True)
    action: Mapped[str] = mapped_column(String(50))
    reviewer: Mapped[str] = mapped_column(String(100), default='system')
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    remap_target: Mapped[str | None] = mapped_column(String(200), nullable=True)
    resolution_scope: Mapped[str] = mapped_column(String(80))
    review_context_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    semantic_signature_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    applicability_policy_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    applicability_fingerprint: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    terminates_future_authority: Mapped[bool] = mapped_column(default=False)
    supersedes_review_group_decision_id: Mapped[int | None] = mapped_column(ForeignKey('review_group_decisions.review_group_decision_id'), nullable=True, index=True)
    terminated_by_review_group_decision_id: Mapped[int | None] = mapped_column(ForeignKey('review_group_decisions.review_group_decision_id'), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    review_group: Mapped[ReviewGroup] = relationship(back_populates='decisions')
