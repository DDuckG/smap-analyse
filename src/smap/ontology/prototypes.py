from __future__ import annotations
from dataclasses import dataclass, field, replace
from smap.canonicalization.alias import AliasRegistry, normalize_alias
from smap.ontology.models import EntitySeed, OntologyRegistry
from smap.providers.base import EmbeddingProvider, EmbeddingPurpose

def _unique_strings(values):
    seen = set()
    unique = []
    for value in values:
        cleaned = ' '.join(value.split()).strip()
        if not cleaned:
            continue
        folded = cleaned.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        unique.append(cleaned)
    return tuple(unique)

def _normalize_compact(text):
    return normalize_alias(text).replace(' ', '')

@dataclass(frozen=True, slots=True)
class EntityPrototypeBundle:
    canonical_entity_id: str
    label: str
    entity_type: str
    entity_kind: str
    knowledge_layer: str
    target_eligible: bool
    aliases: tuple[str, ...]
    normalized_aliases: tuple[str, ...]
    compact_aliases: tuple[str, ...]
    related_phrases: tuple[str, ...]
    domain_anchor_phrases: tuple[str, ...]
    neighboring_entity_ids: tuple[str, ...]
    neighboring_entity_labels: tuple[str, ...]
    neighboring_aspect_ids: tuple[str, ...]
    neighboring_aspect_labels: tuple[str, ...]
    neighboring_issue_ids: tuple[str, ...]
    neighboring_issue_labels: tuple[str, ...]
    related_topic_keys: tuple[str, ...]
    related_topic_labels: tuple[str, ...]
    anti_confusion_phrases: tuple[str, ...]
    taxonomy_ids: tuple[str, ...]
    prototype_text: str
    alias_lookup_text: str
    vector: tuple[float, ...] | None = None

@dataclass(frozen=True, slots=True)
class TopicPrototypeBundle:
    topic_key: str
    label: str
    knowledge_layer: str
    topic_family: str
    main_path_policy: str
    seed_phrases: tuple[str, ...]
    negative_phrases: tuple[str, ...]
    domain_tags: tuple[str, ...]
    excluded_domain_tags: tuple[str, ...]
    issue_centric: bool
    reportable: bool
    related_entity_ids: tuple[str, ...]
    related_aspect_ids: tuple[str, ...]
    related_issue_ids: tuple[str, ...]
    compatible_entity_types: tuple[str, ...]
    prototype_text: str
    vector: tuple[float, ...] | None = None

@dataclass(frozen=True, slots=True)
class AspectPrototypeBundle:
    aspect_id: str
    label: str
    seed_phrases: tuple[str, ...]
    negative_phrases: tuple[str, ...]
    compatible_entity_types: tuple[str, ...]
    linked_topic_keys: tuple[str, ...]
    related_issue_ids: tuple[str, ...]
    prototype_text: str
    vector: tuple[float, ...] | None = None

@dataclass(frozen=True, slots=True)
class IssuePrototypeBundle:
    issue_id: str
    label: str
    seed_phrases: tuple[str, ...]
    negative_phrases: tuple[str, ...]
    compatible_entity_types: tuple[str, ...]
    linked_topic_keys: tuple[str, ...]
    related_aspect_ids: tuple[str, ...]
    prototype_text: str
    vector: tuple[float, ...] | None = None

@dataclass(slots=True)
class PrototypeRegistry:
    ontology: OntologyRegistry
    embedding_provider: EmbeddingProvider | None = None
    alias_registry: AliasRegistry = field(init=False)
    entities: dict[str, EntityPrototypeBundle] = field(init=False, default_factory=dict)
    topics: dict[str, TopicPrototypeBundle] = field(init=False, default_factory=dict)
    aspects: dict[str, AspectPrototypeBundle] = field(init=False, default_factory=dict)
    issues: dict[str, IssuePrototypeBundle] = field(init=False, default_factory=dict)
    phrase_lexicon: tuple[str, ...] = field(init=False, default_factory=tuple)
    taxonomy_labels: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.alias_registry = AliasRegistry.from_ontology(self.ontology)
        self.taxonomy_labels = {node.id: node.label for node in self.ontology.taxonomy_nodes}
        self.entities = self._build_entities()
        self.aspects = self._build_aspects()
        self.issues = self._build_issues()
        self.topics = self._build_topics()
        self.phrase_lexicon = collect_phrase_lexicon(self.ontology)
        self._precompute_vectors()

    def aspect_seed_map(self):
        return {bundle.aspect_id: bundle.seed_phrases for bundle in self.aspects.values()}

    def issue_seed_map(self):
        return {bundle.issue_id: bundle.seed_phrases for bundle in self.issues.values()}

    def eligible_target_keys(self):
        return {bundle.canonical_entity_id for bundle in self.entities.values() if bundle.target_eligible}

    def eligible_concept_keys(self):
        return {bundle.canonical_entity_id for bundle in self.entities.values() if bundle.entity_kind == 'concept' and bundle.target_eligible}

    def topic_catalog_label(self, topic_key):
        bundle = self.topics.get(topic_key)
        return bundle.label if bundle is not None else topic_key.replace('_', ' ').title()

    def _build_entities(self):
        bundles: dict[str, EntityPrototypeBundle] = {}
        for entity in self.ontology.entities:
            bundles[entity.id] = self._entity_bundle(entity)
        return bundles

    def _entity_bundle(self, entity):
        aliases = _unique_strings([entity.name, *entity.aliases, *entity.compact_aliases])
        normalized_aliases = _unique_strings([normalize_alias(alias) for alias in aliases])
        compact_aliases = _unique_strings([_normalize_compact(alias) for alias in aliases])
        taxonomy_labels = [self.taxonomy_labels.get(taxonomy_id, taxonomy_id) for taxonomy_id in entity.taxonomy_ids]
        entity_labels = {item.id: item.name for item in self.ontology.entities}
        topic_labels = {topic.topic_key: topic.label for topic in self.ontology.topics}
        aspect_labels = {aspect.id: aspect.label for aspect in self.ontology.aspect_categories}
        issue_labels = {issue.id: issue.label for issue in self.ontology.issue_categories}
        related_topics = [topic for topic in self.ontology.topics if entity.id in topic.related_entity_ids]
        neighboring_entity_ids = _unique_strings([entity_id for topic in related_topics for entity_id in topic.related_entity_ids if entity_id != entity.id] + entity.neighboring_entity_ids)
        neighboring_entity_labels = _unique_strings([entity_labels[item] for item in neighboring_entity_ids if item in entity_labels])
        neighboring_aspect_ids = _unique_strings([aspect_id for topic in related_topics for aspect_id in topic.related_aspect_ids] + entity.neighboring_aspect_ids)
        neighboring_aspect_labels = _unique_strings([aspect_labels[item] for item in neighboring_aspect_ids if item in aspect_labels])
        neighboring_issue_ids = _unique_strings([issue_id for topic in related_topics for issue_id in topic.related_issue_ids])
        neighboring_issue_labels = _unique_strings([issue_labels[item] for item in neighboring_issue_ids if item in issue_labels])
        related_topic_keys = _unique_strings([topic.topic_key for topic in related_topics])
        related_topic_labels = _unique_strings([topic_labels[item] for item in related_topic_keys if item in topic_labels])
        domain_anchor_phrases = _unique_strings([*entity.domain_anchor_phrases, *[phrase for topic in related_topics for phrase in topic.seed_phrases[:4]], *related_topic_labels])
        related_phrases = _unique_strings([*entity.related_phrases, *taxonomy_labels, entity.entity_type, *neighboring_aspect_labels, *neighboring_issue_labels])
        anti_confusion = _unique_strings(entity.anti_confusion_phrases)
        prototype_parts = [entity.name, entity.description or '', entity.entity_type, *aliases, *related_phrases, *domain_anchor_phrases, *neighboring_entity_labels, *related_topic_labels, *anti_confusion]
        alias_lookup_parts = [entity.name, *aliases, *entity.compact_aliases]
        return EntityPrototypeBundle(canonical_entity_id=entity.id, label=entity.name, entity_type=entity.entity_type, entity_kind=entity.entity_kind, knowledge_layer=entity.knowledge_layer, target_eligible=entity.target_eligible, aliases=aliases, normalized_aliases=normalized_aliases, compact_aliases=compact_aliases, related_phrases=related_phrases, domain_anchor_phrases=domain_anchor_phrases, neighboring_entity_ids=neighboring_entity_ids, neighboring_entity_labels=neighboring_entity_labels, neighboring_aspect_ids=neighboring_aspect_ids, neighboring_aspect_labels=neighboring_aspect_labels, neighboring_issue_ids=neighboring_issue_ids, neighboring_issue_labels=neighboring_issue_labels, related_topic_keys=related_topic_keys, related_topic_labels=related_topic_labels, anti_confusion_phrases=anti_confusion, taxonomy_ids=tuple(entity.taxonomy_ids), prototype_text=' || '.join(_unique_strings(prototype_parts)), alias_lookup_text=' || '.join(_unique_strings(alias_lookup_parts)))

    def _build_aspects(self):
        topic_labels = {topic.topic_key: topic.label for topic in self.ontology.topics}
        issues = {issue.id: issue.label for issue in self.ontology.issue_categories}
        bundles: dict[str, AspectPrototypeBundle] = {}
        for category in self.ontology.aspect_categories:
            related_issues = [issues[item] for item in category.related_issue_ids if item in issues]
            linked_topics = [topic_labels[item] for item in category.linked_topic_keys if item in topic_labels]
            prototype_text = ' || '.join(_unique_strings([category.label, category.description or '', *category.seed_phrases, *related_issues, *linked_topics, *category.compatible_entity_types]))
            bundles[category.id] = AspectPrototypeBundle(aspect_id=category.id, label=category.label, seed_phrases=_unique_strings(category.seed_phrases), negative_phrases=_unique_strings(category.negative_phrases), compatible_entity_types=tuple(category.compatible_entity_types), linked_topic_keys=tuple(category.linked_topic_keys), related_issue_ids=tuple(category.related_issue_ids), prototype_text=prototype_text)
        return bundles

    def _build_issues(self):
        topic_labels = {topic.topic_key: topic.label for topic in self.ontology.topics}
        aspects = {aspect.id: aspect.label for aspect in self.ontology.aspect_categories}
        bundles: dict[str, IssuePrototypeBundle] = {}
        for category in self.ontology.issue_categories:
            related_aspects = [aspects[item] for item in category.related_aspect_ids if item in aspects]
            linked_topics = [topic_labels[item] for item in category.linked_topic_keys if item in topic_labels]
            prototype_text = ' || '.join(_unique_strings([category.label, category.description or '', *category.seed_phrases, *related_aspects, *linked_topics, *category.compatible_entity_types]))
            bundles[category.id] = IssuePrototypeBundle(issue_id=category.id, label=category.label, seed_phrases=_unique_strings(category.seed_phrases), negative_phrases=_unique_strings(category.negative_phrases), compatible_entity_types=tuple(category.compatible_entity_types), linked_topic_keys=tuple(category.linked_topic_keys), related_aspect_ids=tuple(category.related_aspect_ids), prototype_text=prototype_text)
        return bundles

    def _build_topics(self):
        entity_labels = {entity.id: entity.name for entity in self.ontology.entities}
        aspect_labels = {aspect.id: aspect.label for aspect in self.ontology.aspect_categories}
        issue_labels = {issue.id: issue.label for issue in self.ontology.issue_categories}
        bundles: dict[str, TopicPrototypeBundle] = {}
        for topic in self.ontology.topics:
            linked_entities = [entity_labels[item] for item in topic.related_entity_ids if item in entity_labels]
            linked_aspects = [aspect_labels[item] for item in topic.related_aspect_ids if item in aspect_labels]
            linked_issues = [issue_labels[item] for item in topic.related_issue_ids if item in issue_labels]
            prototype_text = ' || '.join(_unique_strings([topic.label, topic.description or '', *topic.seed_phrases, *topic.domain_tags, *linked_entities, *linked_aspects, *linked_issues, *topic.compatible_entity_types]))
            bundles[topic.topic_key] = TopicPrototypeBundle(topic_key=topic.topic_key, label=topic.label, knowledge_layer=topic.knowledge_layer, topic_family=topic.topic_family or topic.topic_key, main_path_policy=topic.main_path_policy, seed_phrases=_unique_strings(topic.seed_phrases), negative_phrases=_unique_strings(topic.negative_phrases), domain_tags=tuple(topic.domain_tags), excluded_domain_tags=tuple(topic.excluded_domain_tags), issue_centric=topic.issue_centric, reportable=topic.reportable, related_entity_ids=tuple(topic.related_entity_ids), related_aspect_ids=tuple(topic.related_aspect_ids), related_issue_ids=tuple(topic.related_issue_ids), compatible_entity_types=tuple(topic.compatible_entity_types), prototype_text=prototype_text)
        return bundles

    def _precompute_vectors(self):
        if self.embedding_provider is None:
            return
        entity_ids = list(self.entities)
        entity_vectors = self.embedding_provider.embed_texts([self.entities[entity_id].prototype_text for entity_id in entity_ids], purpose=EmbeddingPurpose.LINKING)
        for entity_id, vector in zip(entity_ids, entity_vectors, strict=True):
            entity_bundle = self.entities[entity_id]
            self.entities[entity_id] = replace(entity_bundle, vector=vector)
        topic_keys = list(self.topics)
        topic_vectors = self.embedding_provider.embed_texts([self.topics[topic_key].prototype_text for topic_key in topic_keys], purpose=EmbeddingPurpose.CLUSTERING)
        for topic_key, vector in zip(topic_keys, topic_vectors, strict=True):
            topic_bundle = self.topics[topic_key]
            self.topics[topic_key] = replace(topic_bundle, vector=vector)
        aspect_ids = list(self.aspects)
        aspect_vectors = self.embedding_provider.embed_texts([self.aspects[aspect_id].prototype_text for aspect_id in aspect_ids], purpose=EmbeddingPurpose.LINKING)
        for aspect_id, vector in zip(aspect_ids, aspect_vectors, strict=True):
            aspect_bundle = self.aspects[aspect_id]
            self.aspects[aspect_id] = replace(aspect_bundle, vector=vector)
        issue_ids = list(self.issues)
        issue_vectors = self.embedding_provider.embed_texts([self.issues[issue_id].prototype_text for issue_id in issue_ids], purpose=EmbeddingPurpose.LINKING)
        for issue_id, vector in zip(issue_ids, issue_vectors, strict=True):
            issue_bundle = self.issues[issue_id]
            self.issues[issue_id] = replace(issue_bundle, vector=vector)

def collect_phrase_lexicon(ontology):
    phrases = []
    for entity in ontology.entities:
        phrases.extend([entity.name, *entity.aliases, *entity.compact_aliases, *entity.related_phrases])
    for topic in ontology.topics:
        phrases.extend([topic.label, *topic.seed_phrases])
    for aspect in ontology.aspect_categories:
        phrases.extend([aspect.label, *aspect.seed_phrases])
    for issue in ontology.issue_categories:
        phrases.extend([issue.label, *issue.seed_phrases])
    scored = sorted({normalize_alias(phrase) for phrase in phrases if len(normalize_alias(phrase).split()) >= 2 or any((character.isdigit() for character in normalize_alias(phrase)))}, key=lambda item: (-len(item.split()), -len(item), item))
    return tuple(scored)
