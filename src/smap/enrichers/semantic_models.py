from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum
from pydantic import BaseModel

class AnchorType(StrEnum):
    TARGET = 'target_anchor'
    ASPECT = 'aspect_anchor'
    ISSUE = 'issue_anchor'
    POLARITY = 'polarity_cue_anchor'
    NEGATION = 'negation_cue'
    UNCERTAINTY = 'uncertainty_cue'
    COMPARISON = 'comparison_cue'
    CONTRAST = 'contrast_cue'
    ESCALATION = 'escalation_cue'
    HEARSAY = 'hearsay_cue'

class EvidenceMode(StrEnum):
    DIRECT_COMPLAINT = 'direct_complaint'
    DIRECT_OBSERVATION = 'direct_observation'
    QUESTION_OR_UNCERTAINTY = 'question_or_uncertainty'
    HEARSAY_OR_RUMOR = 'hearsay_or_rumor'
    COMPARISON_BASED_CRITIQUE = 'comparison_based_critique'
    ESCALATION_SIGNAL = 'escalation_signal'

class EvidenceScope(StrEnum):
    LOCAL = 'local'
    INHERITED = 'inherited'
    AMBIGUOUS = 'ambiguous'

class EvidenceSpan(BaseModel):
    start: int
    end: int
    text: str
    anchor_type: str | None = None
    label: str | None = None
    segment_id: str | None = None

class ScoreComponent(BaseModel):
    name: str
    value: float
    reason: str

@dataclass(slots=True)
class SemanticSegment:
    segment_id: str
    start: int
    end: int
    text: str
    normalized_text: str
    leading_boundary: str | None = None
    contrastive: bool = False
    question_like: bool = False

@dataclass(slots=True)
class SemanticAnchor:
    anchor_type: AnchorType
    label: str
    normalized_text: str
    start: int
    end: int
    text: str
    source: str
    confidence: float
    segment_id: str
    polarity: float | None = None
    metadata: dict[str, str | bool | float | list[str]] = field(default_factory=dict)

    def to_evidence_span(self):
        return EvidenceSpan(start=self.start, end=self.end, text=self.text, anchor_type=self.anchor_type.value, label=self.label, segment_id=self.segment_id)

@dataclass(slots=True)
class TargetReference:
    target_key: str
    target_text: str
    canonical_entity_id: str | None
    entity_type: str | None
    concept_entity_id: str | None = None
    unresolved_cluster_id: str | None = None
    target_kind: str = 'surface'
    grounding_confidence: float = 0.0
    inherited: bool = False
    inherited_from_mention_id: str | None = None

@dataclass(slots=True)
class MentionSentimentHypothesis:
    mention_id: str
    sentiment: str
    confidence: float
    score: float
    evidence_spans: list[EvidenceSpan]
    score_components: list[ScoreComponent]
    uncertainty_flags: list[str]
    segment_ids: list[str]
    semantic_routing: str = 'semantic_full'
    corroboration_confidence: float | None = None

@dataclass(slots=True)
class TargetSentimentHypothesis:
    mention_id: str
    target: TargetReference
    sentiment: str
    confidence: float
    score: float
    evidence_scope: EvidenceScope
    evidence_spans: list[EvidenceSpan]
    score_components: list[ScoreComponent]
    uncertainty_flags: list[str]
    segment_ids: list[str]
    semantic_routing: str = 'semantic_full'
    target_grounding_confidence: float | None = None
    corroboration_confidence: float | None = None

@dataclass(slots=True)
class AspectOpinionHypothesis:
    mention_id: str
    aspect: str
    target: TargetReference | None
    sentiment: str
    confidence: float
    evidence_scope: EvidenceScope
    evidence_spans: list[EvidenceSpan]
    score_components: list[ScoreComponent]
    uncertainty_flags: list[str]
    segment_id: str
    semantic_routing: str = 'semantic_full'
    target_grounding_confidence: float | None = None
    corroboration_confidence: float | None = None

@dataclass(slots=True)
class IssueSignalHypothesis:
    mention_id: str
    issue_category: str
    target: TargetReference | None
    severity: str
    evidence_mode: EvidenceMode
    confidence: float
    evidence_scope: EvidenceScope
    evidence_spans: list[EvidenceSpan]
    score_components: list[ScoreComponent]
    uncertainty_flags: list[str]
    segment_id: str
    semantic_routing: str = 'semantic_full'
    target_grounding_confidence: float | None = None
    corroboration_confidence: float | None = None
    corroboration_count: int = 1
