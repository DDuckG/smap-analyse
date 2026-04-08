from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from smap.core.types import utc_now


class LabelStudioResult(BaseModel):
    from_name: str
    to_name: str
    type: str
    value: dict[str, Any]


class LabelStudioPrediction(BaseModel):
    model_version: str
    score: float | None = None
    result: list[LabelStudioResult] = Field(default_factory=list)


class LabelStudioAnnotation(BaseModel):
    id: int | str | None = None
    completed_by: dict[str, Any] | None = None
    result: list[LabelStudioResult] = Field(default_factory=list)


class LabelStudioTask(BaseModel):
    id: int | str | None = None
    data: dict[str, Any]
    predictions: list[LabelStudioPrediction] = Field(default_factory=list)
    annotations: list[LabelStudioAnnotation] = Field(default_factory=list)


class ImportedSemanticAnnotation(BaseModel):
    mention_id: str
    source_uap_id: str | None = None
    task_type: str
    semantic_kind: str
    segment_id: str | None = None
    evidence_span_signature: str | None = None
    source_target_key: str | None = None
    source_aspect: str | None = None
    source_issue_category: str | None = None
    source_evidence_mode: str | None = None
    source_severity: str | None = None
    target_key: str | None = None
    aspect: str | None = None
    issue_category: str | None = None
    evidence_mode: str | None = None
    severity: str | None = None
    evidence_text: str | None = None
    notes: str | None = None
    provenance_task_id: str
    annotation_id: str
    reviewer: str
    prediction_model_versions: list[str] = Field(default_factory=list)
    imported_at: datetime = Field(default_factory=utc_now)


class ImportedTopicAnnotation(BaseModel):
    topic_key: str
    source_topic_label: str
    approved_topic_label: str
    topic_signature: str | None = None
    topic_term_signature: str | None = None
    topic_profile_signature: str | None = None
    merge_into_topic_key: str | None = None
    usefulness_judgment: str | None = None
    stability_judgment: str | None = None
    top_terms: list[str] = Field(default_factory=list)
    representative_document_ids: list[str] = Field(default_factory=list)
    notes: str | None = None
    provenance_task_id: str
    annotation_id: str
    reviewer: str
    prediction_model_versions: list[str] = Field(default_factory=list)
    imported_at: datetime = Field(default_factory=utc_now)
