from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, cast
from smap.canonicalization.alias import normalize_alias
from smap.enrichers.models import AspectOpinionFact, EntityFact, FactProvenance, IssueSignalFact, TopicArtifactFact, TopicFact
from smap.enrichers.segmentation import SemanticSegmenter
from smap.enrichers.topic_artifacts import build_topic_artifact_evidence
from smap.enrichers.topic_quality import assess_topic_artifact, assess_topic_label_health, build_topic_profile_signature, build_topic_run_id, build_topic_signature, build_topic_term_signature, emerging_topic_flag, growth_delta_for_topic, load_latest_topic_snapshot, select_reporting_topic_label, topic_stability_score
from smap.hil.feedback_store import PromotedTopicLineageRecord
from smap.hil.topic_lineage import apply_topic_lineage
from smap.ml.heads import BinaryLinearHead, MulticlassLinearHead
from smap.normalization.models import MentionRecord
from smap.ontology.prototypes import PrototypeRegistry
from smap.providers.base import EmbeddingProvider, EmbeddingPurpose, TopicDocument, TopicProvider
from smap.providers.fallback import FallbackTopicProvider
from smap.threads.models import MentionContext

class TopicCandidateEnricher:
    name = 'topic'

    def __init__(self, topic_provider=None, *, artifact_history_dir=None, reviewed_topic_lineage=None, embedding_provider=None, prototype_registry=None, artifact_role_head=None, lineage_head=None):
        self.topic_provider = topic_provider or FallbackTopicProvider()
        self.segmenter = SemanticSegmenter()
        self._topic_facts_by_mention: dict[str, list[TopicFact]] = defaultdict(list)
        self._topic_artifacts: list[TopicArtifactFact] = []
        self.artifact_history_dir = artifact_history_dir
        self.reviewed_topic_lineage = reviewed_topic_lineage or []
        self.embedding_provider = embedding_provider
        self.prototype_registry = prototype_registry
        self.artifact_role_head = artifact_role_head
        self.lineage_head = lineage_head

    def prepare(self, mentions, contexts, *, entity_facts=None, aspect_facts=None, issue_facts=None):
        del contexts
        entity_facts = entity_facts or []
        aspect_facts = list(aspect_facts or [])
        issue_facts = list(issue_facts or [])
        entities_by_mention: dict[str, dict[str, list[str]]] = defaultdict(lambda: {'canonical': [], 'concept': []})
        aspects_by_mention: dict[str, list[str]] = defaultdict(list)
        issues_by_mention: dict[str, list[str]] = defaultdict(list)
        for entity_fact in entity_facts:
            if entity_fact.canonical_entity_id is not None:
                entities_by_mention[entity_fact.mention_id]['canonical'].append(entity_fact.canonical_entity_id)
            if entity_fact.concept_entity_id is not None:
                entities_by_mention[entity_fact.mention_id]['concept'].append(entity_fact.concept_entity_id)
        for aspect_fact in aspect_facts:
            aspects_by_mention[aspect_fact.mention_id].append(aspect_fact.aspect)
        for issue_fact in issue_facts:
            issues_by_mention[issue_fact.mention_id].append(issue_fact.issue_category)
        documents: list[TopicDocument] = []
        mention_by_document: dict[str, MentionRecord] = {}
        document_by_id: dict[str, TopicDocument] = {}
        for mention in mentions:
            mention_entities = entities_by_mention.get(mention.mention_id, {'canonical': [], 'concept': []})
            topic_hint_text = self._topic_hint_text(mention)
            segments = self.segmenter.segment(mention.raw_text)
            if not segments:
                documents.append(TopicDocument(document_id=f'{mention.mention_id}:whole', mention_id=mention.mention_id, source_uap_id=mention.source_uap_id, text=mention.raw_text, normalized_text=self._merge_topic_text(mention.normalized_text_compact, topic_hint_text), metadata={'text_quality_label': mention.text_quality_label, 'text_quality_flags': list(mention.text_quality_flags), 'semantic_route_hint': mention.semantic_route_hint, 'canonical_entity_ids': list(dict.fromkeys(mention_entities['canonical'])), 'concept_entity_ids': list(dict.fromkeys(mention_entities['concept'])), 'aspect_ids': list(dict.fromkeys(aspects_by_mention.get(mention.mention_id, []))), 'issue_ids': list(dict.fromkeys(issues_by_mention.get(mention.mention_id, []))), 'topic_hint_text': topic_hint_text or None}))
                mention_by_document[f'{mention.mention_id}:whole'] = mention
                document_by_id[f'{mention.mention_id}:whole'] = documents[-1]
                continue
            for segment in segments:
                document_id = f'{mention.mention_id}:{segment.segment_id}'
                documents.append(TopicDocument(document_id=document_id, mention_id=mention.mention_id, source_uap_id=mention.source_uap_id, text=segment.text, normalized_text=self._merge_topic_text(segment.normalized_text, topic_hint_text if segment.segment_id == 'seg-0' else ''), segment_id=segment.segment_id, metadata={'text_quality_label': mention.text_quality_label, 'text_quality_flags': list(mention.text_quality_flags), 'semantic_route_hint': mention.semantic_route_hint, 'canonical_entity_ids': list(dict.fromkeys(mention_entities['canonical'])), 'concept_entity_ids': list(dict.fromkeys(mention_entities['concept'])), 'aspect_ids': list(dict.fromkeys(aspects_by_mention.get(mention.mention_id, []))), 'issue_ids': list(dict.fromkeys(issues_by_mention.get(mention.mention_id, []))), 'topic_hint_text': topic_hint_text or None}))
                mention_by_document[document_id] = mention
                document_by_id[document_id] = documents[-1]
        clustering_embeddings = None
        text_embedding_cache: dict[str, tuple[float, ...]] = {}
        if self.embedding_provider is not None and documents:
            document_texts = [document.normalized_text or document.text for document in documents]
            clustering_embeddings = self.embedding_provider.embed_texts(document_texts, purpose=EmbeddingPurpose.CLUSTERING)
            text_embedding_cache.update(zip(document_texts, clustering_embeddings, strict=True))
        discovery = self.topic_provider.discover(documents, embeddings=clustering_embeddings)
        global_document_frequency = self._topic_document_frequency(documents)
        documents_by_topic: dict[str, list[TopicDocument]] = defaultdict(list)
        self._topic_facts_by_mention = defaultdict(list)
        pending_assignments: list[tuple[MentionRecord, str | None, str, str, float]] = []
        for assignment in discovery.assignments:
            matched_mention = mention_by_document.get(assignment.document_id)
            if matched_mention is None:
                continue
            document = document_by_id.get(assignment.document_id)
            if document is not None:
                documents_by_topic[assignment.topic_key].append(document)
            segment_id = assignment.document_id.split(':', 1)[1] if ':' in assignment.document_id else None
            pending_assignments.append((matched_mention, segment_id, assignment.topic_key, assignment.topic_label, assignment.confidence))
        previous_snapshot = load_latest_topic_snapshot(self.artifact_history_dir / 'latest_topic_artifacts.jsonl') if self.artifact_history_dir is not None else []
        time_window_start = min((mention.posted_at.isoformat() for mention in mentions if mention.posted_at is not None), default=None)
        time_window_end = max((mention.posted_at.isoformat() for mention in mentions if mention.posted_at is not None), default=None)
        run_id = build_topic_run_id(provider_name=self.topic_provider.provenance.provider_name, model_id=self.topic_provider.provenance.model_id, document_ids=[document.document_id for document in documents])
        artifact_facts: list[TopicArtifactFact] = []
        for artifact in discovery.artifacts:
            topic_documents = documents_by_topic.get(artifact.topic_key, [])
            prototype_bundle = self.prototype_registry.topics.get(artifact.topic_key) if self.prototype_registry is not None else None
            artifact_evidence = build_topic_artifact_evidence(topic_bundle=prototype_bundle, documents=topic_documents, entity_facts=entity_facts, aspect_facts=aspect_facts, issue_facts=issue_facts, prototype_registry=self.prototype_registry, embedding_provider=self.embedding_provider, text_embedding_cache=text_embedding_cache, role_head=self.artifact_role_head)
            canonical_evidence_terms = list(artifact_evidence.canonical_evidence_phrases)
            evidence_terms = canonical_evidence_terms or list(artifact.top_terms)
            topic_signature = build_topic_signature(provider_name=artifact.provider_provenance.provider_name, model_id=artifact.provider_provenance.model_id, top_terms=evidence_terms, representative_document_ids=artifact.representative_document_ids)
            topic_term_signature = build_topic_term_signature(evidence_terms)
            topic_profile_signature = build_topic_profile_signature(topic_key=artifact.topic_key, topic_label=artifact.topic_label, topic_family=prototype_bundle.topic_family if prototype_bundle is not None else artifact.topic_key, top_canonical_entity_ids=artifact_evidence.top_canonical_entity_ids, aspect_profile=artifact_evidence.aspect_profile, issue_profile=artifact_evidence.issue_profile, canonical_evidence_phrases=canonical_evidence_terms or evidence_terms)
            quality = assess_topic_artifact(artifact, topic_documents, canonical_evidence_phrases=evidence_terms, supporting_phrases=list(artifact_evidence.supporting_phrases), embedding_provider=self.embedding_provider, global_document_frequency=global_document_frequency, text_embedding_cache=text_embedding_cache)
            artifact_reporting_status = cast(Literal['reportable', 'discovery_only', 'suppressed'], quality.reporting_status)
            topic_fact = TopicArtifactFact(topic_key=artifact.topic_key, topic_label=artifact.topic_label, topic_family=prototype_bundle.topic_family if prototype_bundle is not None else artifact.topic_key, effective_topic_key=artifact.topic_key, effective_topic_label=artifact.topic_label, topic_size=artifact.topic_size, top_terms=evidence_terms, salient_terms=list(quality.salient_terms), canonical_evidence_phrases=canonical_evidence_terms, canonical_evidence_details=list(artifact_evidence.canonical_evidence_details), supporting_phrases=list(artifact_evidence.supporting_phrases), supporting_phrase_details=list(artifact_evidence.supporting_phrase_details), issue_supporting_phrases=list(artifact_evidence.issue_supporting_phrases), aspect_supporting_phrases=list(artifact_evidence.aspect_supporting_phrases), top_canonical_entity_ids=list(artifact_evidence.top_canonical_entity_ids), aspect_profile=list(artifact_evidence.aspect_profile), issue_profile=list(artifact_evidence.issue_profile), representative_document_ids=list(artifact.representative_document_ids), representative_texts=[document.text for document in topic_documents[:3]], artifact_version='topic-artifact-v5', topic_signature=topic_signature, topic_term_signature=topic_term_signature, topic_profile_signature=topic_profile_signature, run_id=run_id, provider_name=artifact.provider_provenance.provider_name, provider_model_id=artifact.provider_provenance.model_id, quality_score=quality.quality_score, usefulness_score=quality.usefulness_score, reportability_score=quality.reportability_score, reportability_reason_flags=list(quality.reason_flags), reporting_status=artifact_reporting_status, chatter_burden=quality.chatter_burden, reaction_only_burden=quality.reaction_only_burden, low_information_burden=quality.low_information_burden, evidence_density=quality.evidence_density, recurrence_score=quality.recurrence_score, business_salience_score=quality.business_salience_score, artifact_purity_score=artifact_evidence.artifact_purity_score, evidence_coherence_score=artifact_evidence.evidence_coherence_score, issue_leak_rate=artifact_evidence.issue_leak_rate, quality_reason_flags=list(quality.reason_flags), weak_topic=quality.weak_topic, noisy_topic=quality.noisy_topic, hashtag_only=quality.hashtag_only, time_window_start=artifact.time_window_start or time_window_start, time_window_end=artifact.time_window_end or time_window_end, provenance=FactProvenance(source_uap_id='topic-artifact', mention_id=artifact.representative_document_ids[0] if artifact.representative_document_ids else artifact.topic_key, provider_version=artifact.provider_provenance.provider_version, rule_version=artifact.provider_provenance.provider_name, evidence_text='; '.join(evidence_terms or list(artifact.top_terms))))
            topic_fact = apply_topic_lineage(topic_fact, self.reviewed_topic_lineage)
            stability_score, previous_match = topic_stability_score(topic_fact, previous_snapshot, embedding_provider=self.embedding_provider, text_embedding_cache=text_embedding_cache, continuity_head=self.lineage_head)
            topic_fact = topic_fact.model_copy(update={'stability_score': stability_score, 'growth_delta': growth_delta_for_topic(topic_fact, previous_match)})
            topic_fact = topic_fact.model_copy(update={'emerging_topic': emerging_topic_flag(topic_fact)})
            label_choice = select_reporting_topic_label(topic_fact, previous_match=previous_match, documents=topic_documents, embedding_provider=self.embedding_provider, global_document_frequency=global_document_frequency, text_embedding_cache=text_embedding_cache)
            topic_fact = topic_fact.model_copy(update={'reporting_topic_key': label_choice.reporting_topic_key, 'reporting_topic_label': label_choice.reporting_topic_label, 'label_source': label_choice.label_source, 'label_confidence': label_choice.label_confidence})
            label_health = assess_topic_label_health(topic_fact.reporting_topic_label, topic_fact)
            updated_reason_flags = list(topic_fact.reportability_reason_flags)
            reporting_status = topic_fact.reporting_status
            if label_health.score < 0.72:
                updated_reason_flags.append('dirty_reporting_label')
                if reporting_status == 'reportable':
                    reporting_status = 'discovery_only'
            if (topic_fact.artifact_purity_score or 0.0) < 0.52:
                updated_reason_flags.append('low_artifact_purity')
                if reporting_status == 'reportable':
                    reporting_status = 'discovery_only'
            if (topic_fact.issue_leak_rate or 0.0) >= 0.34 and (not (prototype_bundle.issue_centric if prototype_bundle is not None else False)):
                updated_reason_flags.append('high_issue_leakage')
                if reporting_status == 'reportable':
                    reporting_status = 'discovery_only'
            artifact_facts.append(topic_fact)
            artifact_facts[-1] = artifact_facts[-1].model_copy(update={'label_health_score': label_health.score, 'label_health_flags': list(label_health.flags), 'reporting_status': reporting_status, 'reportability_reason_flags': sorted(set(updated_reason_flags))})
        self._topic_artifacts = artifact_facts
        effective_identity_by_topic_key = {artifact.topic_key: (artifact.effective_topic_key or artifact.topic_key, artifact.effective_topic_label or artifact.topic_label, artifact.reporting_topic_key or artifact.effective_topic_key or artifact.topic_key, artifact.reporting_topic_label or artifact.effective_topic_label or artifact.topic_label) for artifact in artifact_facts}
        for matched_mention, segment_id, topic_key, topic_label, confidence in pending_assignments:
            effective_topic_key, effective_topic_label, reporting_topic_key, reporting_topic_label = effective_identity_by_topic_key.get(topic_key, (topic_key, topic_label, topic_key, topic_label))
            mention_reporting_status: Literal['reportable', 'discovery_only', 'suppressed'] = next((artifact.reporting_status for artifact in artifact_facts if artifact.topic_key == topic_key), 'reportable')
            self._topic_facts_by_mention[matched_mention.mention_id].append(TopicFact(mention_id=matched_mention.mention_id, source_uap_id=matched_mention.source_uap_id, topic_key=topic_key, topic_label=topic_label, effective_topic_key=effective_topic_key, effective_topic_label=effective_topic_label, reporting_topic_key=reporting_topic_key, reporting_topic_label=reporting_topic_label, reporting_status=mention_reporting_status, label_source=next((artifact.label_source for artifact in artifact_facts if artifact.topic_key == topic_key), None), label_confidence=next((artifact.label_confidence for artifact in artifact_facts if artifact.topic_key == topic_key), None), label_health_score=next((artifact.label_health_score for artifact in artifact_facts if artifact.topic_key == topic_key), None), label_health_flags=next((list(artifact.label_health_flags) for artifact in artifact_facts if artifact.topic_key == topic_key), []), reportability_score=next((artifact.reportability_score for artifact in artifact_facts if artifact.topic_key == topic_key), None), reportability_reason_flags=next((list(artifact.reportability_reason_flags) for artifact in artifact_facts if artifact.topic_key == topic_key), []), confidence=confidence, segment_id=segment_id, provenance=FactProvenance(source_uap_id=matched_mention.source_uap_id, mention_id=matched_mention.mention_id, provider_version=self.topic_provider.version, rule_version='topic-provider-v3', evidence_text=matched_mention.raw_text)))

    def enrich(self, mention, context):
        del context
        return list(self._topic_facts_by_mention.get(mention.mention_id, []))

    def artifacts(self):
        return list(self._topic_artifacts)

    def _topic_document_frequency(self, documents):
        document_frequency: Counter[str] = Counter()
        for document in documents:
            tokens = {token for token in document.normalized_text.split() if len(token) >= 4 and (not token.startswith('#'))}
            document_frequency.update(tokens)
        return dict(document_frequency)

    def _topic_hint_text(self, mention):
        if mention.summary_title is None:
            return ''
        normalized_hint = normalize_alias(mention.summary_title)
        if not normalized_hint:
            return ''
        if normalized_hint in mention.normalized_text_compact:
            return ''
        return normalized_hint

    def _merge_topic_text(self, base_text, hint_text):
        merged_parts = [part.strip() for part in (base_text, hint_text) if part.strip()]
        return ' '.join(dict.fromkeys(merged_parts))
