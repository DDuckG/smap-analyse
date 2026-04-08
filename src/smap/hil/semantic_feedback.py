from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from smap.canonicalization.alias import normalize_alias
from smap.enrichers.models import AspectOpinionFact, IssueSignalFact
from smap.enrichers.semantic_models import ScoreComponent
from smap.hil.feedback_store import PromotedSemanticKnowledgeRecord


@dataclass(frozen=True, slots=True)
class SemanticFeedbackApplicationResult:
    aspect_facts: list[AspectOpinionFact]
    issue_facts: list[IssueSignalFact]


@dataclass(frozen=True, slots=True)
class SemanticFeedbackMatch:
    record: PromotedSemanticKnowledgeRecord
    matched_dimensions: tuple[str, ...]


def apply_promoted_semantic_feedback(
    *,
    aspect_facts: list[AspectOpinionFact],
    issue_facts: list[IssueSignalFact],
    promoted_records: list[PromotedSemanticKnowledgeRecord],
) -> SemanticFeedbackApplicationResult:
    if not promoted_records:
        return SemanticFeedbackApplicationResult(
            aspect_facts=list(aspect_facts),
            issue_facts=list(issue_facts),
        )
    return SemanticFeedbackApplicationResult(
        aspect_facts=[_apply_aspect_feedback(fact, promoted_records) for fact in aspect_facts],
        issue_facts=[_apply_issue_feedback(fact, promoted_records) for fact in issue_facts],
    )


def semantic_region_key_for_aspect(fact: AspectOpinionFact) -> str:
    return _semantic_region_key(
        semantic_kind="aspect_opinion",
        segment_id=fact.segment_id,
        evidence_span_signature=_evidence_span_signature(fact.evidence_spans),
        label=fact.aspect,
        target_key=fact.target_key,
        evidence_text=_aspect_evidence_text(fact),
    )


def semantic_region_key_for_issue(fact: IssueSignalFact) -> str:
    return _semantic_region_key(
        semantic_kind="issue_signal",
        segment_id=fact.segment_id,
        evidence_span_signature=_evidence_span_signature(fact.evidence_spans),
        label=fact.issue_category,
        target_key=fact.target_key,
        evidence_text=_issue_evidence_text(fact),
    )


def _apply_aspect_feedback(
    fact: AspectOpinionFact,
    promoted_records: list[PromotedSemanticKnowledgeRecord],
) -> AspectOpinionFact:
    match = _best_record_match(
        promoted_records,
        semantic_kind="aspect_opinion",
        mention_id=fact.mention_id,
        exact_key=semantic_region_key_for_aspect(fact),
        segment_id=fact.segment_id,
        evidence_span_signature=_evidence_span_signature(fact.evidence_spans),
        normalized_evidence=_normalized_text(_aspect_evidence_text(fact)),
        target_key=fact.target_key,
        source_label=fact.aspect,
    )
    if match is None:
        return fact
    updated_components = [
        *fact.score_components,
        ScoreComponent(
            name="semantic_feedback_override",
            value=0.22,
            reason=(
                f"promoted_semantic_feedback:{match.record.record_id}:"
                f"{'+'.join(match.matched_dimensions)}"
            ),
        ),
    ]
    return fact.model_copy(
        update={
            "aspect": match.record.aspect or fact.aspect,
            "target_key": match.record.target_key or fact.target_key,
            "canonical_entity_id": (
                fact.canonical_entity_id
                if match.record.target_key in {None, fact.target_key}
                else None
            ),
            "score_components": updated_components,
        }
    )


def _apply_issue_feedback(
    fact: IssueSignalFact,
    promoted_records: list[PromotedSemanticKnowledgeRecord],
) -> IssueSignalFact:
    match = _best_record_match(
        promoted_records,
        semantic_kind="issue_signal",
        mention_id=fact.mention_id,
        exact_key=semantic_region_key_for_issue(fact),
        segment_id=fact.segment_id,
        evidence_span_signature=_evidence_span_signature(fact.evidence_spans),
        normalized_evidence=_normalized_text(_issue_evidence_text(fact)),
        target_key=fact.target_key,
        source_label=fact.issue_category,
        source_evidence_mode=str(fact.evidence_mode),
        source_severity=fact.severity,
    )
    if match is None:
        return fact
    updated_components = [
        *fact.score_components,
        ScoreComponent(
            name="semantic_feedback_override",
            value=0.22,
            reason=(
                f"promoted_semantic_feedback:{match.record.record_id}:"
                f"{'+'.join(match.matched_dimensions)}"
            ),
        ),
    ]
    return fact.model_copy(
        update={
            "issue_category": match.record.issue_category or fact.issue_category,
            "evidence_mode": match.record.evidence_mode or fact.evidence_mode,
            "severity": match.record.severity or fact.severity,
            "target_key": match.record.target_key or fact.target_key,
            "canonical_entity_id": (
                fact.canonical_entity_id
                if match.record.target_key in {None, fact.target_key}
                else None
            ),
            "score_components": updated_components,
        }
    )


def _best_record_match(
    records: list[PromotedSemanticKnowledgeRecord],
    *,
    semantic_kind: str,
    mention_id: str,
    exact_key: str,
    segment_id: str | None,
    evidence_span_signature: str | None,
    normalized_evidence: str,
    target_key: str | None,
    source_label: str | None,
    source_evidence_mode: str | None = None,
    source_severity: str | None = None,
) -> SemanticFeedbackMatch | None:
    ranked = sorted(records, key=lambda item: (item.promoted_at, item.record_id), reverse=True)
    best: SemanticFeedbackMatch | None = None
    best_rank: tuple[int, int, str, str] | None = None
    for record in ranked:
        if record.semantic_kind != semantic_kind:
            continue
        matched_dimensions: list[str] = []
        if record.semantic_region_key == exact_key:
            matched_dimensions.append("semantic_region_key")
        else:
            segment_match = bool(record.segment_id and segment_id and record.segment_id == segment_id)
            mention_match = record.mention_id == mention_id
            span_match = bool(
                record.evidence_span_signature
                and evidence_span_signature
                and record.evidence_span_signature == evidence_span_signature
            )
            evidence_match = bool(
                record.normalized_evidence_text
                and record.normalized_evidence_text == normalized_evidence
            )
            target_match = _optional_match(record.source_target_key or record.target_key, target_key)
            label_match = _optional_match(_record_source_label(record), source_label)
            evidence_mode_match = _optional_match(record.source_evidence_mode, source_evidence_mode)
            severity_match = _optional_match(record.source_severity, source_severity)
            if segment_match:
                matched_dimensions.append("segment_id")
            if span_match:
                matched_dimensions.append("evidence_span_signature")
            if evidence_match:
                matched_dimensions.append("normalized_evidence_text")
            if record.source_target_key is not None and target_match:
                matched_dimensions.append("source_target_key")
            if _record_source_label(record) is not None and label_match:
                matched_dimensions.append("source_label")
            if record.source_evidence_mode is not None and evidence_mode_match:
                matched_dimensions.append("source_evidence_mode")
            if record.source_severity is not None and severity_match:
                matched_dimensions.append("source_severity")
            safe_local_fallback = (
                mention_match
                and segment_match
                and label_match
                and target_match
                and (record.source_evidence_mode is None or evidence_mode_match)
                and (record.source_severity is None or severity_match)
            )
            if safe_local_fallback:
                matched_dimensions.append("mention_id")
            if not matched_dimensions:
                continue
            if record.segment_id is not None and not segment_match:
                continue
            if record.evidence_span_signature is not None and not span_match:
                continue
            if _record_source_label(record) is not None and not label_match:
                continue
            if record.source_target_key is not None and not target_match:
                continue
            if record.source_evidence_mode is not None and not evidence_mode_match:
                continue
            if record.source_severity is not None and not severity_match:
                continue
            if record.normalized_evidence_text and not evidence_match and not safe_local_fallback:
                continue
            if not any(
                dimension in matched_dimensions
                for dimension in {"semantic_region_key", "evidence_span_signature", "normalized_evidence_text", "mention_id"}
            ):
                continue
        rank = (
            1 if "semantic_region_key" in matched_dimensions else 0,
            len(matched_dimensions),
            record.promoted_at,
            record.record_id,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = SemanticFeedbackMatch(record=record, matched_dimensions=tuple(matched_dimensions))
    return best


def _aspect_evidence_text(fact: AspectOpinionFact) -> str:
    if fact.opinion_text.strip():
        return fact.opinion_text
    if fact.evidence_spans:
        return " ".join(span.text for span in fact.evidence_spans if span.text.strip())
    return fact.provenance.evidence_text


def _issue_evidence_text(fact: IssueSignalFact) -> str:
    if fact.evidence_spans:
        return " ".join(span.text for span in fact.evidence_spans if span.text.strip())
    return fact.provenance.evidence_text


def _semantic_region_key(
    *,
    semantic_kind: str,
    segment_id: str | None,
    evidence_span_signature: str | None,
    label: str | None,
    target_key: str | None,
    evidence_text: str | None,
) -> str:
    parts = [
        semantic_kind,
        normalize_alias(segment_id or ""),
        evidence_span_signature or "",
        normalize_alias(label or ""),
        normalize_alias(target_key or ""),
        _normalized_text(evidence_text),
    ]
    return "|".join(part for part in parts if part)


def _normalized_text(value: str | None) -> str:
    return normalize_alias(value or "")


def _optional_match(expected: str | None, actual: str | None) -> bool:
    if expected is None:
        return True
    return normalize_alias(expected) == normalize_alias(actual or "")


def _record_source_label(record: PromotedSemanticKnowledgeRecord) -> str | None:
    return record.source_aspect or record.source_issue_category


def _evidence_span_signature(spans: Sequence[object]) -> str | None:
    normalized: list[str] = []
    for span in spans:
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        segment_id = getattr(span, "segment_id", None)
        if start is None or end is None:
            continue
        normalized.append(f"{segment_id or ''}:{int(start)}-{int(end)}")
    if not normalized:
        return None
    return "|".join(sorted(normalized))
