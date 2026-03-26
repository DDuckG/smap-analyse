from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field
from smap.canonicalization.models import DiscoveryMethod
from smap.core.types import utc_now
from smap.enrichers.semantic_models import EvidenceMode, EvidenceScope, EvidenceSpan, ScoreComponent
SentimentLabel = Literal['positive', 'negative', 'neutral', 'mixed']
StanceLabel = Literal['support', 'oppose', 'question', 'neutral']
IssueSeverity = Literal['low', 'medium', 'high', 'critical_like_proxy']
InfluenceTier = Literal['nano', 'micro', 'mid', 'macro']

class FactProvenance(BaseModel):
    source_uap_id: str
    mention_id: str
    provider_version: str
    rule_version: str
    evidence_text: str
    evidence_span: tuple[int, int] | None = None
    generated_at: datetime = Field(default_factory=utc_now)

class EntityFact(BaseModel):
    mention_id: str
    source_uap_id: str
    candidate_text: str
    canonical_entity_id: str | None = None
    concept_entity_id: str | None = None
    entity_type: str | None = None
    confidence: float
    matched_by: str
    resolution_kind: Literal['canonical_entity', 'concept', 'unresolved_candidate'] = 'unresolved_candidate'
    resolved_entity_kind: Literal['entity', 'concept'] | None = None
    knowledge_layer: Literal['base', 'domain', 'domain_overlay', 'review_overlay', 'batch_local_candidate'] | None = None
    target_eligible: bool = False
    unresolved_cluster_id: str | None = None
    unresolved_cluster_size: int = 0
    surface_specificity: float = 0.0
    unresolved_reason: str | None = None
    canonical_candidate_ids: list[str] = Field(default_factory=list)
    discovered_by: list[DiscoveryMethod] = Field(default_factory=list)
    provenance: FactProvenance

class EntityCandidateClusterFact(BaseModel):
    cluster_id: str
    normalized_surface: str
    representative_surface: str
    knowledge_layer: Literal['batch_local_candidate'] = 'batch_local_candidate'
    entity_type_hint: str | None = None
    mention_count: int
    source_languages: list[str] = Field(default_factory=list)
    discovered_by: list[DiscoveryMethod] = Field(default_factory=list)
    representative_mention_ids: list[str] = Field(default_factory=list)
    candidate_canonical_ids: list[str] = Field(default_factory=list)
    promotion_state: Literal['reviewable', 'promotable', 'promoted'] = 'reviewable'
    provenance: FactProvenance

class KeywordFact(BaseModel):
    mention_id: str
    source_uap_id: str
    keyphrase: str
    confidence: float
    provenance: FactProvenance

class TopicFact(BaseModel):
    mention_id: str
    source_uap_id: str
    topic_key: str
    topic_label: str
    effective_topic_key: str | None = None
    effective_topic_label: str | None = None
    reporting_topic_key: str | None = None
    reporting_topic_label: str | None = None
    reporting_status: Literal['reportable', 'discovery_only', 'suppressed'] = 'reportable'
    label_source: str | None = None
    label_confidence: float | None = None
    label_health_score: float | None = None
    label_health_flags: list[str] = Field(default_factory=list)
    reportability_score: float | None = None
    reportability_reason_flags: list[str] = Field(default_factory=list)
    confidence: float
    segment_id: str | None = None
    provenance: FactProvenance

class TopicEvidencePhrase(BaseModel):
    phrase: str
    role: Literal['canonical', 'supporting', 'issue_support', 'aspect_support', 'debug_only']
    source_types: list[str] = Field(default_factory=list)
    purity_score: float
    readability_score: float
    semantic_score: float | None = None
    support_score: float | None = None
    provenance_flags: list[str] = Field(default_factory=list)

class TopicArtifactFact(BaseModel):
    topic_key: str
    topic_label: str
    topic_family: str | None = None
    effective_topic_key: str | None = None
    effective_topic_label: str | None = None
    reporting_topic_key: str | None = None
    reporting_topic_label: str | None = None
    label_source: str | None = None
    label_confidence: float | None = None
    label_health_score: float | None = None
    label_health_flags: list[str] = Field(default_factory=list)
    topic_size: int
    top_terms: list[str] = Field(default_factory=list)
    salient_terms: list[str] = Field(default_factory=list)
    canonical_evidence_phrases: list[str] = Field(default_factory=list)
    canonical_evidence_details: list[TopicEvidencePhrase] = Field(default_factory=list)
    supporting_phrases: list[str] = Field(default_factory=list)
    supporting_phrase_details: list[TopicEvidencePhrase] = Field(default_factory=list)
    issue_supporting_phrases: list[str] = Field(default_factory=list)
    aspect_supporting_phrases: list[str] = Field(default_factory=list)
    top_canonical_entity_ids: list[str] = Field(default_factory=list)
    aspect_profile: list[str] = Field(default_factory=list)
    issue_profile: list[str] = Field(default_factory=list)
    representative_document_ids: list[str] = Field(default_factory=list)
    representative_texts: list[str] = Field(default_factory=list)
    artifact_version: str = 'topic-artifact-v5'
    topic_signature: str | None = None
    topic_term_signature: str | None = None
    topic_profile_signature: str | None = None
    run_id: str | None = None
    provider_name: str | None = None
    provider_model_id: str | None = None
    growth_delta: float | None = None
    stability_score: float | None = None
    quality_score: float | None = None
    usefulness_score: float | None = None
    reportability_score: float | None = None
    reportability_reason_flags: list[str] = Field(default_factory=list)
    reporting_status: Literal['reportable', 'discovery_only', 'suppressed'] = 'reportable'
    chatter_burden: float | None = None
    reaction_only_burden: float | None = None
    low_information_burden: float | None = None
    evidence_density: float | None = None
    recurrence_score: float | None = None
    business_salience_score: float | None = None
    artifact_purity_score: float | None = None
    evidence_coherence_score: float | None = None
    issue_leak_rate: float | None = None
    quality_reason_flags: list[str] = Field(default_factory=list)
    weak_topic: bool = False
    noisy_topic: bool = False
    hashtag_only: bool = False
    emerging_topic: bool = False
    reviewed_topic_id: str | None = None
    topic_lineage_id: str | None = None
    reviewed_merge_target_key: str | None = None
    topic_review_resolution: str | None = None
    reviewed_label_override: str | None = None
    reviewed_usefulness_judgment: str | None = None
    reviewed_stability_judgment: str | None = None
    time_window_start: str | None = None
    time_window_end: str | None = None
    provenance: FactProvenance

class SentimentFact(BaseModel):
    mention_id: str
    source_uap_id: str
    sentiment: SentimentLabel
    score: float
    confidence: float
    semantic_routing: Literal['semantic_lite', 'semantic_full'] = 'semantic_full'
    sentiment_confidence: float | None = None
    corroboration_confidence: float | None = None
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    score_components: list[ScoreComponent] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    provenance: FactProvenance

class TargetSentimentFact(BaseModel):
    mention_id: str
    source_uap_id: str
    target_key: str
    target_text: str
    canonical_entity_id: str | None = None
    concept_entity_id: str | None = None
    unresolved_cluster_id: str | None = None
    target_kind: Literal['canonical_entity', 'concept', 'unresolved_cluster', 'surface'] = 'surface'
    entity_type: str | None = None
    sentiment: SentimentLabel
    score: float
    confidence: float
    semantic_routing: Literal['semantic_lite', 'semantic_full'] = 'semantic_full'
    sentiment_confidence: float | None = None
    target_grounding_confidence: float | None = None
    corroboration_confidence: float | None = None
    target_inherited: bool = False
    inherited_from_mention_id: str | None = None
    evidence_scope: EvidenceScope = EvidenceScope.LOCAL
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    score_components: list[ScoreComponent] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    provenance: FactProvenance

class StanceFact(BaseModel):
    mention_id: str
    source_uap_id: str
    stance: StanceLabel
    confidence: float
    provenance: FactProvenance

class IntentFact(BaseModel):
    mention_id: str
    source_uap_id: str
    intent: str
    confidence: float
    provenance: FactProvenance

class AspectOpinionFact(BaseModel):
    mention_id: str
    source_uap_id: str
    aspect: str
    opinion_text: str
    sentiment: str
    confidence: float
    target_key: str | None = None
    canonical_entity_id: str | None = None
    concept_entity_id: str | None = None
    unresolved_cluster_id: str | None = None
    target_kind: Literal['canonical_entity', 'concept', 'unresolved_cluster', 'surface'] = 'surface'
    semantic_routing: Literal['semantic_lite', 'semantic_full'] = 'semantic_full'
    sentiment_confidence: float | None = None
    target_grounding_confidence: float | None = None
    corroboration_confidence: float | None = None
    target_inherited: bool = False
    inherited_from_mention_id: str | None = None
    evidence_scope: EvidenceScope = EvidenceScope.LOCAL
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    score_components: list[ScoreComponent] = Field(default_factory=list)
    segment_id: str | None = None
    provenance: FactProvenance

class IssueSignalFact(BaseModel):
    mention_id: str
    source_uap_id: str
    issue_category: str
    severity: IssueSeverity
    confidence: float
    evidence_mode: EvidenceMode
    target_key: str | None = None
    canonical_entity_id: str | None = None
    concept_entity_id: str | None = None
    unresolved_cluster_id: str | None = None
    target_kind: Literal['canonical_entity', 'concept', 'unresolved_cluster', 'surface'] = 'surface'
    semantic_routing: Literal['semantic_lite', 'semantic_full'] = 'semantic_full'
    issue_evidence_confidence: float | None = None
    target_grounding_confidence: float | None = None
    corroboration_confidence: float | None = None
    target_inherited: bool = False
    inherited_from_mention_id: str | None = None
    evidence_scope: EvidenceScope = EvidenceScope.LOCAL
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    score_components: list[ScoreComponent] = Field(default_factory=list)
    corroboration_count: int = 1
    segment_id: str | None = None
    provenance: FactProvenance

class SourceInfluenceFact(BaseModel):
    mention_id: str
    source_uap_id: str
    author_id: str
    channel: str
    influence_tier: InfluenceTier
    engagement_score: float
    confidence: float
    provenance: FactProvenance

class EnrichmentBundle(BaseModel):
    entity_facts: list[EntityFact] = Field(default_factory=list)
    entity_candidate_clusters: list[EntityCandidateClusterFact] = Field(default_factory=list)
    keyword_facts: list[KeywordFact] = Field(default_factory=list)
    topic_facts: list[TopicFact] = Field(default_factory=list)
    topic_artifacts: list[TopicArtifactFact] = Field(default_factory=list)
    sentiment_facts: list[SentimentFact] = Field(default_factory=list)
    target_sentiment_facts: list[TargetSentimentFact] = Field(default_factory=list)
    stance_facts: list[StanceFact] = Field(default_factory=list)
    intent_facts: list[IntentFact] = Field(default_factory=list)
    aspect_opinion_facts: list[AspectOpinionFact] = Field(default_factory=list)
    issue_signal_facts: list[IssueSignalFact] = Field(default_factory=list)
    source_influence_facts: list[SourceInfluenceFact] = Field(default_factory=list)
