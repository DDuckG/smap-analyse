from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from smap.ontology.models import OntologyRegistry, TopicDefinition
from smap.providers.base import EmbeddingProvider, EmbeddingPurpose, ProviderMetadata, ProviderProvenance, TopicArtifact, TopicAssignment, TopicDiscoveryResult, TopicDocument, TopicProvider
_GENERIC_TOPIC_TERMS = {'ban', 'cho', 'co', 'cua', 'for', 'haha', 'hehe', 'khong', 'la', 'nha', 'nay', 'nhe', 'one', 'on', 'roi', 'the', 'thi', 'this', 'thoi', 'voi'}

@dataclass(slots=True)
class _TopicPrototype:
    definition: TopicDefinition
    prototype_text: str

class OntologyGuidedTopicProvider(TopicProvider):

    def __init__(self, *, ontology, embedding_provider=None, secondary_unmatched_topic_key='topic.misc_unclassified'):
        self.ontology = ontology
        self.embedding_provider = embedding_provider
        self.secondary_unmatched_topic_key = secondary_unmatched_topic_key
        self.version = 'ontology-topic-v3'
        self.provenance = ProviderProvenance(provider_kind='topic', provider_name='ontology_topic', provider_version='ontology-topic-v3', model_id=embedding_provider.provenance.model_id if embedding_provider is not None else 'ontology-only', device=embedding_provider.provenance.device if embedding_provider is not None else 'cpu')
        self._prototypes = [_TopicPrototype(definition=topic, prototype_text=' '.join((part for part in [topic.label, topic.description or '', *topic.seed_phrases] if part))) for topic in ontology.topics]
        self._prototype_embedding_cache: dict[str, tuple[float, ...]] | None = None
        self._domain_id = ontology.domain_id or ontology.metadata.name
        self._family_has_domain_primary = {topic.topic_family or topic.topic_key for topic in ontology.topics if topic.main_path_policy == 'domain_primary'}
        self._entity_type_by_id = {entity.id: entity.entity_type for entity in ontology.entities}

    def discover(self, documents, *, embeddings=None):
        if not documents:
            return TopicDiscoveryResult(assignments=[], artifacts=[])
        prototype_embeddings = self._prototype_embeddings()
        assignments: list[TopicAssignment] = []
        docs_by_topic: dict[str, list[TopicDocument]] = defaultdict(list)
        for index, document in enumerate(documents):
            topic_match = self._assign_document(document, embedding=embeddings[index] if embeddings is not None and index < len(embeddings) else None, prototype_embeddings=prototype_embeddings)
            if topic_match is None:
                continue
            topic_definition, score, metadata = topic_match
            assignments.append(TopicAssignment(document_id=document.document_id, topic_key=topic_definition.topic_key, topic_label=topic_definition.label, confidence=round(score, 3), representative=False, metadata=metadata))
            docs_by_topic[topic_definition.topic_key].append(document)
        artifacts: list[TopicArtifact] = []
        definitions_by_key = {topic.topic_key: topic for topic in self.ontology.topics}
        for topic_key, topic_docs in sorted(docs_by_topic.items()):
            topic_definition = definitions_by_key[topic_key]
            top_terms = self._top_terms(topic_docs, topic_definition)
            artifacts.append(TopicArtifact(topic_key=topic_definition.topic_key, topic_label=topic_definition.label, top_terms=tuple(top_terms), representative_document_ids=tuple((document.document_id for document in topic_docs[:3])), topic_size=len(topic_docs), provider_provenance=self.provenance, metadata={'mode': 'ontology_guided', 'knowledge_layer': topic_definition.knowledge_layer}))
        return TopicDiscoveryResult(assignments=assignments, artifacts=artifacts)

    def _assign_document(self, document, *, embedding, prototype_embeddings):
        normalized = document.normalized_text.casefold().strip()
        if not normalized:
            return None
        canonical_entity_ids = document.metadata.get('canonical_entity_ids')
        entity_ids = {str(item) for item in canonical_entity_ids if isinstance(item, str)} if isinstance(canonical_entity_ids, list) else set()
        aspect_metadata = document.metadata.get('aspect_ids')
        aspect_ids = {str(item) for item in aspect_metadata if isinstance(item, str)} if isinstance(aspect_metadata, list) else set()
        issue_metadata = document.metadata.get('issue_ids')
        issue_ids = {str(item) for item in issue_metadata if isinstance(item, str)} if isinstance(issue_metadata, list) else set()
        document_entity_types = {self._entity_type_by_id[item] for item in entity_ids if item in self._entity_type_by_id}
        domain_grounded = bool(entity_ids or aspect_ids or issue_ids)
        scored: list[tuple[TopicDefinition, float, dict[str, ProviderMetadata]]] = []
        for prototype in self._prototypes:
            if not prototype.definition.reportable:
                continue
            lexical, matched_seed_phrases = self._lexical_score(normalized, prototype.definition)
            if lexical <= 0.0 and (not entity_ids) and (not aspect_ids) and (not issue_ids):
                continue
            semantic = self._semantic_score(normalized, prototype, embedding=embedding, prototype_embeddings=prototype_embeddings)
            entity_bonus = 0.12 if entity_ids & set(prototype.definition.related_entity_ids) else 0.0
            aspect_bonus = 0.08 if aspect_ids & set(prototype.definition.related_aspect_ids) else 0.0
            issue_bonus = 0.08 if issue_ids & set(prototype.definition.related_issue_ids) else 0.0
            negative_penalty = 0.18 if self._contains_any(normalized, prototype.definition.negative_phrases) else 0.0
            domain_profile_eligible = self._topic_matches_active_domain(prototype.definition, document_entity_types=document_entity_types, domain_grounded=domain_grounded, lexical=lexical, entity_bonus=entity_bonus, aspect_bonus=aspect_bonus, issue_bonus=issue_bonus)
            domain_primary_bonus = 0.1 if prototype.definition.main_path_policy == 'domain_primary' and (lexical >= 0.2 or entity_bonus > 0.0 or aspect_bonus > 0.0 or (issue_bonus > 0.0)) else 0.0
            family_penalty = 0.0
            if self._topic_family_is_domain_owned(prototype.definition) and prototype.definition.main_path_policy != 'domain_primary':
                if lexical < 0.78 and entity_bonus == 0.0 and (aspect_bonus == 0.0) and (issue_bonus == 0.0):
                    family_penalty += 0.24
                elif lexical < 0.68 and entity_bonus == 0.0 and (aspect_bonus == 0.0):
                    family_penalty += 0.12
            issue_leak_penalty = 0.14 if issue_bonus > 0.0 and (not prototype.definition.issue_centric) and (aspect_bonus == 0.0) and (entity_bonus == 0.0) and (lexical < 0.42) else 0.0
            compatibility_bonus = 0.06 if document_entity_types and prototype.definition.compatible_entity_types and document_entity_types & set(prototype.definition.compatible_entity_types) else 0.0
            if not self._topic_is_main_path_eligible(prototype.definition, domain_grounded=domain_grounded, lexical=lexical, entity_bonus=entity_bonus, aspect_bonus=aspect_bonus, issue_bonus=issue_bonus, document_entity_types=document_entity_types, domain_profile_eligible=domain_profile_eligible):
                continue
            score = lexical * 0.54 + semantic * 0.2 + entity_bonus + aspect_bonus + issue_bonus + domain_primary_bonus + compatibility_bonus - negative_penalty - family_penalty - issue_leak_penalty
            if lexical > 0.0 and semantic > 0.0:
                score += 0.08
            if lexical >= 0.45:
                score += 0.08
            if score >= 0.32:
                scored.append((prototype.definition, min(score, 0.98), {'assignment_mode': 'ontology_guided', 'lexical_score': round(lexical, 4), 'semantic_score': round(semantic, 4), 'entity_bonus': round(entity_bonus, 4), 'aspect_bonus': round(aspect_bonus, 4), 'issue_bonus': round(issue_bonus, 4), 'domain_primary_bonus': round(domain_primary_bonus, 4), 'compatibility_bonus': round(compatibility_bonus, 4), 'family_penalty': round(family_penalty, 4), 'issue_leak_penalty': round(issue_leak_penalty, 4), 'domain_profile_eligible': domain_profile_eligible, 'matched_seed_phrases': matched_seed_phrases[:4]}))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[1], item[0].topic_key))
        best_by_family: dict[str, tuple[TopicDefinition, float, dict[str, ProviderMetadata]]] = {}
        for item in scored:
            family = item[0].topic_family or item[0].topic_key
            best_by_family.setdefault(family, item)
        ranked = sorted(best_by_family.values(), key=lambda item: (-item[1], item[0].topic_key))
        return ranked[0]

    def _lexical_score(self, normalized_text, topic):
        if not topic.seed_phrases:
            return (0.0, [])
        hits = 0.0
        matched_phrases: list[str] = []
        for phrase in topic.seed_phrases:
            normalized_phrase = phrase.casefold().strip()
            if not normalized_phrase:
                continue
            if normalized_phrase in normalized_text:
                hits += 1.0 if ' ' in normalized_phrase else 0.65
                matched_phrases.append(phrase)
        if hits <= 0.0:
            return (0.0, [])
        lexical_score = min(hits / max(min(len(topic.seed_phrases), 4) * 0.65, 1.0) + 0.18, 1.0)
        return (lexical_score, matched_phrases)

    def _semantic_score(self, normalized_text, prototype, *, embedding, prototype_embeddings):
        if self.embedding_provider is None:
            return 0.0
        topic_embedding = prototype_embeddings.get(prototype.definition.topic_key)
        if topic_embedding is None:
            return 0.0
        query_embedding = embedding if embedding is not None else self.embedding_provider.embed_texts([normalized_text], purpose=EmbeddingPurpose.CLUSTERING)[0]
        return max(sum((a * b for a, b in zip(query_embedding, topic_embedding, strict=True))), 0.0)

    def _prototype_embeddings(self):
        if self.embedding_provider is None or not self._prototypes:
            return {}
        if self._prototype_embedding_cache is not None:
            return self._prototype_embedding_cache
        vectors = self.embedding_provider.embed_texts([prototype.prototype_text for prototype in self._prototypes], purpose=EmbeddingPurpose.CLUSTERING)
        self._prototype_embedding_cache = {prototype.definition.topic_key: vector for prototype, vector in zip(self._prototypes, vectors, strict=True)}
        return self._prototype_embedding_cache

    def _top_terms(self, documents, topic):
        phrase_counts: Counter[str] = Counter()
        for phrase in topic.seed_phrases:
            normalized_phrase = phrase.casefold().strip()
            if not normalized_phrase:
                continue
            for document in documents:
                if normalized_phrase in document.normalized_text.casefold():
                    phrase_counts[phrase] += 1
        if phrase_counts:
            return [phrase for phrase, _ in phrase_counts.most_common(5)]
        term_counts: Counter[str] = Counter()
        for document in documents:
            for token in document.normalized_text.casefold().split():
                if len(token) < 3 or token in _GENERIC_TOPIC_TERMS:
                    continue
                term_counts[token] += 1
        preferred_terms = [phrase for phrase in topic.seed_phrases if phrase and phrase.casefold() in term_counts]
        ranked_terms = preferred_terms + [term for term, _ in term_counts.most_common(8) if term not in preferred_terms]
        return ranked_terms[:5] or [topic.label]

    def _contains_any(self, normalized_text, phrases):
        return any((phrase.casefold().strip() in normalized_text for phrase in phrases if phrase.strip()))

    def _topic_matches_active_domain(self, topic, *, document_entity_types, domain_grounded, lexical, entity_bonus, aspect_bonus, issue_bonus):
        if topic.knowledge_layer == 'domain':
            return True
        if not domain_grounded:
            return True
        if topic.domain_tags and self._domain_id in set(topic.domain_tags):
            return True
        if topic.compatible_entity_types and document_entity_types & set(topic.compatible_entity_types):
            return entity_bonus > 0.0 or aspect_bonus > 0.0 or issue_bonus > 0.0 or (lexical >= 0.72)
        return entity_bonus > 0.0 and lexical >= 0.66

    def _topic_is_main_path_eligible(self, topic, *, domain_grounded, lexical, entity_bonus, aspect_bonus, issue_bonus, document_entity_types, domain_profile_eligible):
        if topic.main_path_policy == 'discovery_only':
            return False
        if topic.excluded_domain_tags and self._domain_id in set(topic.excluded_domain_tags):
            return False
        if topic.knowledge_layer == 'domain':
            return lexical >= 0.18 or entity_bonus > 0.0 or aspect_bonus > 0.0 or (issue_bonus > 0.0)
        if not domain_grounded:
            return lexical >= 0.42 or semantic_bonus_gate(topic=topic, lexical=lexical, entity_bonus=entity_bonus, aspect_bonus=aspect_bonus, issue_bonus=issue_bonus)
        if topic.main_path_policy == 'generic_fallback':
            if self._topic_family_is_domain_owned(topic):
                compatible_type_match = bool(document_entity_types and topic.compatible_entity_types and document_entity_types & set(topic.compatible_entity_types))
                return lexical >= 0.74 and (entity_bonus > 0.0 or aspect_bonus > 0.0 or issue_bonus > 0.0 or (compatible_type_match and lexical >= 0.82))
            return domain_profile_eligible and (lexical >= 0.58 or entity_bonus > 0.0 or aspect_bonus > 0.0 or (issue_bonus > 0.0))
        return domain_profile_eligible

    def _topic_family_is_domain_owned(self, topic):
        return (topic.topic_family or topic.topic_key) in self._family_has_domain_primary

def semantic_bonus_gate(*, topic, lexical, entity_bonus, aspect_bonus, issue_bonus):
    if topic.main_path_policy == 'domain_primary':
        return lexical >= 0.2 or entity_bonus > 0.0 or aspect_bonus > 0.0 or (issue_bonus > 0.0)
    return lexical >= 0.52 or entity_bonus > 0.0 or aspect_bonus > 0.0 or (issue_bonus > 0.0)
