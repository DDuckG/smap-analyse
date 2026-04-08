from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from smap.canonicalization.alias import normalize_alias
from smap.core.settings import Settings
from smap.core.types import utc_now
from smap.hil.label_studio_schemas import ImportedSemanticAnnotation, ImportedTopicAnnotation
from smap.storage.repository import write_jsonl


class ApprovedSemanticAnnotationRecord(ImportedSemanticAnnotation):
    record_id: str
    schema_version: str = "approved-semantic-annotation-v1"


class ApprovedTopicReviewRecord(ImportedTopicAnnotation):
    record_id: str
    schema_version: str = "approved-topic-review-v1"


class PromotedSemanticKnowledgeRecord(BaseModel):
    record_id: str
    promoted_from_record_id: str
    semantic_kind: str
    mention_id: str
    source_uap_id: str | None = None
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
    normalized_evidence_text: str
    semantic_region_key: str
    notes: str | None = None
    reviewer: str
    prediction_model_versions: list[str] = Field(default_factory=list)
    promoted_at: str = ""
    schema_version: str = "promoted-semantic-knowledge-v1"


class BenchmarkPromotedSemanticGoldRecord(BaseModel):
    record_id: str
    promoted_from_record_id: str
    semantic_kind: str
    mention_id: str
    source_uap_id: str | None = None
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
    normalized_evidence_text: str
    semantic_region_key: str
    promoted_at: str = ""
    schema_version: str = "semantic-benchmark-gold-v1"


class PromotedTopicLineageRecord(BaseModel):
    record_id: str
    promoted_from_record_id: str
    source_topic_key: str
    canonical_topic_key: str
    canonical_topic_label: str
    reviewed_topic_id: str
    topic_lineage_id: str
    topic_signature: str | None = None
    topic_term_signature: str | None = None
    topic_profile_signature: str | None = None
    merge_into_topic_key: str | None = None
    usefulness_judgment: str | None = None
    stability_judgment: str | None = None
    notes: str | None = None
    reviewer: str
    promoted_at: str = ""
    schema_version: str = "promoted-topic-lineage-v1"


class FeedbackStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.semantic_path = settings.semantic_feedback_path
        self.topic_path = settings.topic_feedback_path
        self.semantic_knowledge_path = settings.semantic_promoted_path
        self.semantic_benchmark_gold_path = settings.semantic_benchmark_gold_path
        self.topic_lineage_path = settings.topic_lineage_path
        self.settings.ensure_directories()

    def load_semantic_annotations(self) -> list[ApprovedSemanticAnnotationRecord]:
        return _load_records(self.semantic_path, ApprovedSemanticAnnotationRecord)

    def load_topic_reviews(self) -> list[ApprovedTopicReviewRecord]:
        return _load_records(self.topic_path, ApprovedTopicReviewRecord)

    def load_promoted_semantic_knowledge(self) -> list[PromotedSemanticKnowledgeRecord]:
        return _load_records(self.semantic_knowledge_path, PromotedSemanticKnowledgeRecord)

    def load_benchmark_promoted_semantic_gold(self) -> list[BenchmarkPromotedSemanticGoldRecord]:
        return _load_records(self.semantic_benchmark_gold_path, BenchmarkPromotedSemanticGoldRecord)

    def load_topic_lineage(self) -> list[PromotedTopicLineageRecord]:
        return _load_records(self.topic_lineage_path, PromotedTopicLineageRecord)

    def upsert_semantic_annotations(self, records: list[ImportedSemanticAnnotation]) -> int:
        materialized = [
            ApprovedSemanticAnnotationRecord(record_id=_semantic_record_id(record), **record.model_dump(mode="json"))
            for record in records
        ]
        existing = {record.record_id: record for record in self.load_semantic_annotations()}
        for record in materialized:
            existing[record.record_id] = record
        write_jsonl(
            self.semantic_path,
            [record.model_dump(mode="json") for _, record in sorted(existing.items(), key=lambda item: item[0])],
        )
        return len(materialized)

    def upsert_topic_reviews(self, records: list[ImportedTopicAnnotation]) -> int:
        materialized = [
            ApprovedTopicReviewRecord(record_id=_topic_record_id(record), **record.model_dump(mode="json"))
            for record in records
        ]
        existing = {record.record_id: record for record in self.load_topic_reviews()}
        for record in materialized:
            existing[record.record_id] = record
        write_jsonl(
            self.topic_path,
            [record.model_dump(mode="json") for _, record in sorted(existing.items(), key=lambda item: item[0])],
        )
        return len(materialized)

    def promote_semantic_annotations(
        self,
        *,
        record_ids: list[str] | None = None,
        promote_to_benchmark_gold: bool = False,
    ) -> int:
        approved = self.load_semantic_annotations()
        selected = [
            record
            for record in approved
            if record_ids is None or record.record_id in set(record_ids)
        ]
        promoted_records = [
            PromotedSemanticKnowledgeRecord(
                record_id=f"promoted:{record.record_id}",
                promoted_from_record_id=record.record_id,
                semantic_kind=record.semantic_kind,
                mention_id=record.mention_id,
                source_uap_id=record.source_uap_id,
                segment_id=record.segment_id,
                evidence_span_signature=record.evidence_span_signature,
                source_target_key=record.source_target_key,
                source_aspect=record.source_aspect,
                source_issue_category=record.source_issue_category,
                source_evidence_mode=record.source_evidence_mode,
                source_severity=record.source_severity,
                target_key=record.target_key,
                aspect=record.aspect,
                issue_category=record.issue_category,
                evidence_mode=record.evidence_mode,
                severity=record.severity,
                evidence_text=record.evidence_text,
                normalized_evidence_text=_normalized_semantic_feedback_text(record.evidence_text),
                semantic_region_key=_semantic_region_key(record),
                notes=record.notes,
                reviewer=record.reviewer,
                prediction_model_versions=record.prediction_model_versions,
                promoted_at=utc_now().isoformat(),
            )
            for record in selected
        ]
        _upsert_records(self.semantic_knowledge_path, promoted_records, key_attr="record_id")
        if promote_to_benchmark_gold:
            gold_records = [
                BenchmarkPromotedSemanticGoldRecord(
                    record_id=f"gold:{record.record_id}",
                    promoted_from_record_id=record.record_id,
                    semantic_kind=record.semantic_kind,
                    mention_id=record.mention_id,
                    source_uap_id=record.source_uap_id,
                    segment_id=record.segment_id,
                    evidence_span_signature=record.evidence_span_signature,
                    source_target_key=record.source_target_key,
                    source_aspect=record.source_aspect,
                    source_issue_category=record.source_issue_category,
                    source_evidence_mode=record.source_evidence_mode,
                    source_severity=record.source_severity,
                    target_key=record.target_key,
                    aspect=record.aspect,
                    issue_category=record.issue_category,
                    evidence_mode=record.evidence_mode,
                    severity=record.severity,
                    evidence_text=record.evidence_text,
                    normalized_evidence_text=_normalized_semantic_feedback_text(record.evidence_text),
                    semantic_region_key=_semantic_region_key(record),
                    promoted_at=utc_now().isoformat(),
                )
                for record in selected
            ]
            _upsert_records(self.semantic_benchmark_gold_path, gold_records, key_attr="record_id")
        return len(promoted_records)

    def promote_topic_reviews(
        self,
        *,
        record_ids: list[str] | None = None,
    ) -> int:
        approved = self.load_topic_reviews()
        selected = [
            record
            for record in approved
            if record_ids is None or record.record_id in set(record_ids)
        ]
        lineage_records = [
            PromotedTopicLineageRecord(
                record_id=f"lineage:{record.record_id}",
                promoted_from_record_id=record.record_id,
                source_topic_key=record.topic_key,
                canonical_topic_key=record.merge_into_topic_key or record.topic_key,
                canonical_topic_label=record.approved_topic_label,
                reviewed_topic_id=record.merge_into_topic_key or record.topic_key,
                topic_lineage_id=_topic_lineage_id(record),
                topic_signature=record.topic_signature,
                topic_term_signature=record.topic_term_signature,
                topic_profile_signature=record.topic_profile_signature,
                merge_into_topic_key=record.merge_into_topic_key,
                usefulness_judgment=record.usefulness_judgment,
                stability_judgment=record.stability_judgment,
                notes=record.notes,
                reviewer=record.reviewer,
                promoted_at=utc_now().isoformat(),
            )
            for record in selected
        ]
        _upsert_records(self.topic_lineage_path, lineage_records, key_attr="record_id")
        return len(lineage_records)


def match_topic_review(
    records: list[ApprovedTopicReviewRecord],
    *,
    topic_signature: str | None,
    topic_term_signature: str | None,
    topic_key: str,
) -> ApprovedTopicReviewRecord | None:
    if not records:
        return None
    ranked = sorted(records, key=lambda item: (item.imported_at, item.record_id), reverse=True)
    if topic_signature:
        for record in ranked:
            if record.topic_signature == topic_signature:
                return record
    if topic_term_signature:
        for record in ranked:
            if record.topic_term_signature == topic_term_signature:
                return record
    for record in ranked:
        if record.topic_key == topic_key:
            return record
    return None


def match_promoted_topic_lineage(
    records: list[PromotedTopicLineageRecord],
    *,
    topic_profile_signature: str | None,
    topic_signature: str | None,
    topic_term_signature: str | None,
    topic_key: str,
) -> PromotedTopicLineageRecord | None:
    if not records:
        return None
    ranked = sorted(records, key=lambda item: (item.promoted_at, item.record_id), reverse=True)
    if topic_profile_signature:
        for record in ranked:
            if record.topic_profile_signature == topic_profile_signature:
                return record
    if topic_signature:
        for record in ranked:
            if record.topic_signature == topic_signature:
                return record
    if topic_term_signature:
        for record in ranked:
            if record.topic_term_signature == topic_term_signature:
                return record
    for record in ranked:
        if record.source_topic_key == topic_key or record.canonical_topic_key == topic_key:
            return record
    return None


def _load_records[ModelT: BaseModel](path: Path, model_cls: type[ModelT]) -> list[ModelT]:
    if not path.exists():
        return []
    records: list[ModelT] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(model_cls.model_validate(json.loads(line)))
    return records


def _semantic_record_id(record: ImportedSemanticAnnotation) -> str:
    return f"{record.provenance_task_id}:{record.annotation_id}:{record.semantic_kind}:{record.mention_id}"


def _topic_record_id(record: ImportedTopicAnnotation) -> str:
    signature = record.topic_signature or record.topic_term_signature or record.topic_key
    return f"{record.provenance_task_id}:{record.annotation_id}:{signature}"


def _semantic_region_key(record: ImportedSemanticAnnotation) -> str:
    normalized_text = _normalized_semantic_feedback_text(record.evidence_text)
    target_key = normalize_alias(record.source_target_key or record.target_key or "") or ""
    semantic_label = normalize_alias(
        record.source_aspect or record.aspect or record.source_issue_category or record.issue_category or ""
    )
    return "|".join(
        item
        for item in (
            record.semantic_kind,
            normalize_alias(record.segment_id or ""),
            record.evidence_span_signature or "",
            semantic_label,
            target_key,
            normalized_text,
        )
        if item
    )


def _normalized_semantic_feedback_text(value: str | None) -> str:
    return normalize_alias(value or "")


def _topic_lineage_id(record: ImportedTopicAnnotation) -> str:
    raw = record.merge_into_topic_key or record.approved_topic_label or record.topic_term_signature or record.topic_key
    return normalize_alias(raw) or raw


def _upsert_records[ModelT: BaseModel](path: Path, records: list[ModelT], *, key_attr: str) -> None:
    if not records:
        return
    existing = {
        str(getattr(record, key_attr)): record
        for record in _load_records(path, type(records[0]))
    }
    for record in records:
        existing[str(getattr(record, key_attr))] = record
    write_jsonl(
        path,
        [
            record.model_dump(mode="json")
            for _, record in sorted(existing.items(), key=lambda item: item[0])
        ],
    )
