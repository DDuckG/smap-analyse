from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field
DiscoveryMethod = Literal['alias_scan', 'brand_context', 'context_span', 'phobert_ner', 'hashtag', 'handle_pattern', 'product_code', 'repeated_phrase', 'rule_ruler', 'title_span', 'token_ngram']
ResolutionMethod = Literal['boundary_alias', 'compact_alias', 'contextual_alias', 'embedding_similarity', 'embedding_similarity_ranked', 'exact_alias', 'fuzzy_alias', 'normalized_alias', 'unresolved']
ResolutionKind = Literal['canonical_entity', 'concept', 'unresolved_candidate']
KnowledgeLayerKind = Literal['base', 'domain', 'domain_overlay', 'review_overlay', 'batch_local_candidate']
ResolvedEntityKind = Literal['entity', 'concept']

class CanonicalEntity(BaseModel):
    canonical_entity_id: str
    name: str
    entity_type: str
    entity_kind: ResolvedEntityKind = 'entity'
    knowledge_layer: KnowledgeLayerKind = 'base'
    active_linking: bool = True
    target_eligible: bool = True
    taxonomy_ids: list[str] = Field(default_factory=list)
    description: str | None = None
    anti_confusion_phrases: list[str] = Field(default_factory=list)
    source: str = 'ontology'

class EntityAlias(BaseModel):
    alias_id: str
    canonical_entity_id: str
    alias: str
    normalized_alias: str
    source: str = 'ontology'

class EntityCandidate(BaseModel):
    candidate_id: str
    source_uap_id: str
    mention_id: str
    text: str
    normalized_text: str = ''
    start_char: int | None = None
    end_char: int | None = None
    entity_type_hint: str | None = None
    confidence: float = 0.5
    discovered_by: list[DiscoveryMethod] = Field(default_factory=list)
    evidence_mention_ids: list[str] = Field(default_factory=list)
    context_text: str | None = None
    surrounding_text: str | None = None
    full_text: str | None = None

class ResolutionDecision(BaseModel):
    source_uap_id: str
    mention_id: str
    candidate_id: str
    candidate_text: str
    matched_alias: str | None = None
    matched_by: ResolutionMethod
    confidence: float
    evidence_span: tuple[int, int] | None = None
    provider_version: str
    canonical_entity_id: str | None = None
    concept_entity_id: str | None = None
    resolution_kind: ResolutionKind = 'unresolved_candidate'
    resolved_entity_kind: ResolvedEntityKind | None = None
    knowledge_layer: KnowledgeLayerKind | None = None
    candidate_canonical_ids: list[str] = Field(default_factory=list)
    unresolved_reason: str | None = None

class EntityMentionFact(BaseModel):
    mention_id: str
    source_uap_id: str
    candidate_text: str
    canonical_entity_id: str | None = None
    concept_entity_id: str | None = None
    entity_type: str | None = None
    confidence: float
    matched_by: str
    discovered_by: list[DiscoveryMethod] = Field(default_factory=list)
