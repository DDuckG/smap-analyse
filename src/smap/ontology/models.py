from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator
OverlayProvenanceValue = str | int | float | bool | None
KnowledgeLayerKind = Literal['base', 'domain', 'domain_overlay', 'review_overlay', 'batch_local_candidate']
EntityKind = Literal['entity', 'concept']

class OntologyMetadata(BaseModel):
    name: str
    version: str
    description: str

class TaxonomyNode(BaseModel):
    id: str
    label: str
    node_type: str
    parent_id: str | None = None
    description: str | None = None

class CategoryDefinition(BaseModel):
    id: str
    label: str
    description: str | None = None
    seed_phrases: list[str] = Field(default_factory=list)
    negative_phrases: list[str] = Field(default_factory=list)
    compatible_entity_types: list[str] = Field(default_factory=list)
    linked_topic_keys: list[str] = Field(default_factory=list)
    related_issue_ids: list[str] = Field(default_factory=list)
    related_aspect_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, OverlayProvenanceValue] = Field(default_factory=dict)

class TopicDefinition(BaseModel):
    topic_key: str
    label: str
    description: str | None = None
    knowledge_layer: KnowledgeLayerKind = 'base'
    topic_family: str | None = None
    main_path_policy: Literal['core', 'generic_fallback', 'domain_primary', 'discovery_only'] = 'core'
    seed_phrases: list[str] = Field(default_factory=list)
    negative_phrases: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)
    excluded_domain_tags: list[str] = Field(default_factory=list)
    issue_centric: bool = False
    reportable: bool = True
    related_entity_ids: list[str] = Field(default_factory=list)
    related_taxonomy_ids: list[str] = Field(default_factory=list)
    related_aspect_ids: list[str] = Field(default_factory=list)
    related_issue_ids: list[str] = Field(default_factory=list)
    compatible_entity_types: list[str] = Field(default_factory=list)
    metadata: dict[str, OverlayProvenanceValue] = Field(default_factory=dict)

class EntitySeed(BaseModel):
    id: str
    name: str
    entity_type: str
    entity_kind: EntityKind = 'entity'
    knowledge_layer: KnowledgeLayerKind = 'base'
    active_linking: bool = True
    target_eligible: bool = True
    aliases: list[str] = Field(default_factory=list)
    compact_aliases: list[str] = Field(default_factory=list)
    taxonomy_ids: list[str] = Field(default_factory=list)
    description: str | None = None
    related_phrases: list[str] = Field(default_factory=list)
    domain_anchor_phrases: list[str] = Field(default_factory=list)
    neighboring_entity_ids: list[str] = Field(default_factory=list)
    neighboring_aspect_ids: list[str] = Field(default_factory=list)
    anti_confusion_phrases: list[str] = Field(default_factory=list)
    metadata: dict[str, OverlayProvenanceValue] = Field(default_factory=dict)

class SourceChannel(BaseModel):
    id: str
    label: str

class OntologyOverlayMetadata(BaseModel):
    name: str
    version: str
    description: str | None = None
    source: str = 'manual'
    layer_kind: KnowledgeLayerKind = 'domain_overlay'

class OverlayActivationSignal(BaseModel):
    phrase: str
    weight: float = 1.0

class OverlayActivationProfile(BaseModel):
    primary_min_score: float | None = None
    primary_min_matched_records: int | None = None
    secondary_min_score: float | None = None
    secondary_min_matched_records: int | None = None
    signals: list[OverlayActivationSignal] = Field(default_factory=list)

class AliasContribution(BaseModel):
    canonical_entity_id: str
    alias: str
    entity_type: str | None = None
    source: str = 'overlay'
    provenance: dict[str, OverlayProvenanceValue] = Field(default_factory=dict)

class NoiseTerm(BaseModel):
    term: str
    reason: str
    source: str = 'overlay'
    provenance: dict[str, OverlayProvenanceValue] = Field(default_factory=dict)

class OntologyOverlay(BaseModel):
    metadata: OntologyOverlayMetadata
    activation: OverlayActivationProfile | None = None
    entities: list[EntitySeed] = Field(default_factory=list)
    aspect_categories: list[CategoryDefinition] = Field(default_factory=list)
    issue_categories: list[CategoryDefinition] = Field(default_factory=list)
    topics: list[TopicDefinition] = Field(default_factory=list)
    alias_contributions: list[AliasContribution] = Field(default_factory=list)
    noise_terms: list[NoiseTerm] = Field(default_factory=list)

class OntologyRegistry(BaseModel):
    metadata: OntologyMetadata
    domain_id: str | None = None
    activation: OverlayActivationProfile | None = None
    entity_types: list[str]
    taxonomy_nodes: list[TaxonomyNode]
    aspect_categories: list[CategoryDefinition]
    intent_categories: list[CategoryDefinition]
    issue_categories: list[CategoryDefinition]
    source_channels: list[SourceChannel]
    entities: list[EntitySeed] = Field(default_factory=list)
    topics: list[TopicDefinition] = Field(default_factory=list)
    alias_contributions: list[AliasContribution] = Field(default_factory=list)
    noise_terms: list[NoiseTerm] = Field(default_factory=list)
    loaded_overlays: list[OntologyOverlayMetadata] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_registry(self):
        node_ids = {node.id for node in self.taxonomy_nodes}
        aspect_ids = {category.id for category in self.aspect_categories}
        issue_ids = {category.id for category in self.issue_categories}
        entity_ids = {entity.id for entity in self.entities}
        for node in self.taxonomy_nodes:
            if node.parent_id and node.parent_id not in node_ids:
                raise ValueError(f'taxonomy node {node.id} references missing parent {node.parent_id}')
        for entity in self.entities:
            if entity.entity_type not in self.entity_types:
                raise ValueError(f'entity {entity.id} references unknown entity type {entity.entity_type}')
            missing_taxonomy = [item for item in entity.taxonomy_ids if item not in node_ids]
            if missing_taxonomy:
                raise ValueError(f'entity {entity.id} references unknown taxonomy ids {missing_taxonomy}')
            missing_neighbor_entities = [item for item in entity.neighboring_entity_ids if item not in entity_ids]
            if missing_neighbor_entities:
                raise ValueError(f'entity {entity.id} references unknown neighboring entities {missing_neighbor_entities}')
            missing_neighbor_aspects = [item for item in entity.neighboring_aspect_ids if item not in aspect_ids]
            if missing_neighbor_aspects:
                raise ValueError(f'entity {entity.id} references unknown neighboring aspects {missing_neighbor_aspects}')
        for topic in self.topics:
            missing_entity_ids = [item for item in topic.related_entity_ids if item not in entity_ids]
            if missing_entity_ids:
                raise ValueError(f'topic {topic.topic_key} references unknown entities {missing_entity_ids}')
            missing_taxonomy = [item for item in topic.related_taxonomy_ids if item not in node_ids]
            if missing_taxonomy:
                raise ValueError(f'topic {topic.topic_key} references unknown taxonomy ids {missing_taxonomy}')
            missing_aspects = [item for item in topic.related_aspect_ids if item not in aspect_ids]
            if missing_aspects:
                raise ValueError(f'topic {topic.topic_key} references unknown aspects {missing_aspects}')
            missing_issues = [item for item in topic.related_issue_ids if item not in issue_ids]
            if missing_issues:
                raise ValueError(f'topic {topic.topic_key} references unknown issues {missing_issues}')
        for alias in self.alias_contributions:
            if alias.canonical_entity_id not in entity_ids:
                raise ValueError(f'alias contribution for unknown entity {alias.canonical_entity_id}')
            if alias.entity_type is not None and alias.entity_type not in self.entity_types:
                raise ValueError(f'alias contribution references unknown entity type {alias.entity_type}')
        for category in self.aspect_categories:
            missing_topics = [item for item in category.linked_topic_keys if item not in {topic.topic_key for topic in self.topics}]
            if missing_topics:
                raise ValueError(f'aspect {category.id} references unknown topics {missing_topics}')
            missing_issues = [item for item in category.related_issue_ids if item not in issue_ids]
            if missing_issues:
                raise ValueError(f'aspect {category.id} references unknown issues {missing_issues}')
        for category in self.issue_categories:
            missing_topics = [item for item in category.linked_topic_keys if item not in {topic.topic_key for topic in self.topics}]
            if missing_topics:
                raise ValueError(f'issue {category.id} references unknown topics {missing_topics}')
            missing_aspects = [item for item in category.related_aspect_ids if item not in aspect_ids]
            if missing_aspects:
                raise ValueError(f'issue {category.id} references unknown aspects {missing_aspects}')
        return self

    @property
    def taxonomy_node_ids(self):
        return {node.id for node in self.taxonomy_nodes}

    @property
    def aspect_category_ids(self):
        return {category.id for category in self.aspect_categories}

    @property
    def intent_category_ids(self):
        return {category.id for category in self.intent_categories}

    @property
    def issue_category_ids(self):
        return {category.id for category in self.issue_categories}

    @property
    def source_channel_ids(self):
        return {channel.id for channel in self.source_channels}
