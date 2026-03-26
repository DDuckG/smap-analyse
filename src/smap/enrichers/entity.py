from __future__ import annotations
import hashlib
import json
from collections.abc import Callable
from smap.canonicalization.alias import AliasRegistry
from smap.canonicalization.candidate_merge import CandidateMerger
from smap.canonicalization.clustering import cluster_unresolved_entity_facts
from smap.canonicalization.discovery import EntityCandidateDiscoverer
from smap.canonicalization.models import EntityCandidate
from smap.canonicalization.service import CanonicalizationEngine
from smap.enrichers.models import EntityCandidateClusterFact, EntityFact, FactProvenance
from smap.normalization.models import MentionRecord
from smap.ontology.models import OntologyRegistry
from smap.providers.base import NERProvider
from smap.review.context import ReviewContext
from smap.threads.models import MentionContext

class EntityExtractionEnricher:
    name = 'entity_extraction'

    def __init__(self, engine, discoverer=None, ontology_registry=None, other_domain_aliases=None, ner_provider_builder=None, candidate_merger=None):
        self.engine = engine
        self.discoverer = discoverer or EntityCandidateDiscoverer(engine.alias_registry)
        self.ontology_registry = ontology_registry
        self.other_domain_aliases = other_domain_aliases or frozenset()
        self._other_domain_compact = frozenset((alias.replace(' ', '') for alias in self.other_domain_aliases))
        self.ner_provider_builder = ner_provider_builder or (lambda _: [])
        self.candidate_merger = candidate_merger or CandidateMerger(embedding_provider=engine.embedding_provider)
        self._context_by_mention: dict[str, ReviewContext] = {}
        self._scope_by_mention: dict[str, str] = {}
        self._discoverers_by_scope: dict[str, EntityCandidateDiscoverer] = {}
        self._engines_by_scope: dict[str, CanonicalizationEngine] = {}
        self._ner_providers_by_scope: dict[str, list[NERProvider]] = {}
        self._default_ner_providers: list[NERProvider] = self.ner_provider_builder(self.engine.alias_registry)

    def _scope_fingerprint(self, context):
        return hashlib.sha256(json.dumps(context.model_dump(mode='json'), sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()[:16]

    def prepare(self, mentions, contexts):
        if self.ontology_registry is None:
            self.discoverer.prepare(mentions, contexts)
            self._default_ner_providers = self.ner_provider_builder(self.engine.alias_registry)
            return
        context_map = {context.mention_id: context for context in contexts}
        mentions_by_scope: dict[str, list[MentionRecord]] = {}
        contexts_by_scope: dict[str, list[MentionContext]] = {}
        scope_contexts: dict[str, ReviewContext] = {}
        self._context_by_mention = {}
        self._scope_by_mention = {}
        self._discoverers_by_scope = {}
        self._engines_by_scope = {}
        self._ner_providers_by_scope = {}
        for mention in mentions:
            review_context = ReviewContext.from_mention(mention, self.ontology_registry)
            scope_fingerprint = self._scope_fingerprint(review_context)
            self._context_by_mention[mention.mention_id] = review_context
            self._scope_by_mention[mention.mention_id] = scope_fingerprint
            mentions_by_scope.setdefault(scope_fingerprint, []).append(mention)
            scope_contexts[scope_fingerprint] = review_context
            mention_context = context_map.get(mention.mention_id)
            if mention_context is not None:
                contexts_by_scope.setdefault(scope_fingerprint, []).append(mention_context)
        for scope_fingerprint, scoped_mentions in mentions_by_scope.items():
            review_context = scope_contexts[scope_fingerprint]
            alias_registry = AliasRegistry.from_ontology(self.ontology_registry, review_context=review_context)
            engine = CanonicalizationEngine(alias_registry=alias_registry, embedding_provider=self.engine.embedding_provider, vector_index=self.engine.vector_index, fuzzy_threshold=self.engine.fuzzy_threshold, embedding_threshold=self.engine.embedding_threshold, vector_namespace=f'canonical:{scope_fingerprint}')
            discoverer = EntityCandidateDiscoverer(alias_registry)
            discoverer.prepare(scoped_mentions, contexts_by_scope.get(scope_fingerprint, []))
            self._engines_by_scope[scope_fingerprint] = engine
            self._discoverers_by_scope[scope_fingerprint] = discoverer
            self._ner_providers_by_scope[scope_fingerprint] = self.ner_provider_builder(alias_registry)

    def discover_candidates_for_mention(self, mention, context):
        scope_fingerprint = self._scope_by_mention.get(mention.mention_id)
        if scope_fingerprint is None:
            discoverer = self.discoverer
            ner_providers = self._default_ner_providers
        else:
            discoverer = self._discoverers_by_scope.get(scope_fingerprint, self.discoverer)
            ner_providers = self._ner_providers_by_scope.get(scope_fingerprint, self._default_ner_providers)
        base_candidates = discoverer.discover(mention, context)
        recognized_spans = []
        for provider in ner_providers:
            recognized_spans.extend(provider.extract(mention.raw_text, mention_id=mention.mention_id, source_uap_id=mention.source_uap_id))
        return self.candidate_merger.merge(base_candidates=base_candidates, recognized_spans=recognized_spans, mention_id=mention.mention_id, source_uap_id=mention.source_uap_id, context_text=context.context_text if context else None, surrounding_text=mention.raw_text)

    def enrich(self, mention, context):
        scope_fingerprint = self._scope_by_mention.get(mention.mention_id)
        engine = self._engines_by_scope.get(scope_fingerprint, self.engine) if scope_fingerprint else self.engine
        candidates = self.discover_candidates_for_mention(mention, context)
        if not candidates:
            return [self._unresolved_fact(mention, reason=self._no_candidate_reason(mention))]
        resolved_facts: dict[str, EntityFact] = {}
        unresolved_facts: list[tuple[float, EntityFact]] = []
        for candidate in candidates:
            decision = engine.resolve(candidate)
            resolved_entity_id = decision.canonical_entity_id or decision.concept_entity_id
            resolved_entity = engine.alias_registry.entities[resolved_entity_id] if resolved_entity_id else None
            entity_type = resolved_entity.entity_type if resolved_entity else candidate.entity_type_hint
            surface_specificity = engine._surface_specificity(candidate, candidate.normalized_text or candidate.text)
            fact = EntityFact(mention_id=mention.mention_id, source_uap_id=mention.source_uap_id, candidate_text=candidate.text, canonical_entity_id=decision.canonical_entity_id, concept_entity_id=decision.concept_entity_id, entity_type=entity_type, confidence=max(decision.confidence, candidate.confidence if decision.canonical_entity_id else 0.0), matched_by=decision.matched_by, resolution_kind=decision.resolution_kind, resolved_entity_kind=decision.resolved_entity_kind, knowledge_layer=decision.knowledge_layer, target_eligible=resolved_entity.target_eligible if resolved_entity is not None else False, surface_specificity=surface_specificity, unresolved_reason=None if decision.canonical_entity_id is not None or decision.concept_entity_id is not None else self._normalize_unresolved_reason(mention, candidate, decision.unresolved_reason), canonical_candidate_ids=decision.candidate_canonical_ids, discovered_by=candidate.discovered_by, provenance=FactProvenance(source_uap_id=mention.source_uap_id, mention_id=mention.mention_id, provider_version=decision.provider_version, rule_version='entity-discovery-resolution-v4', evidence_text=candidate.surrounding_text or mention.raw_text, evidence_span=decision.evidence_span))
            if fact.canonical_entity_id is not None or fact.concept_entity_id is not None:
                if not self._keep_resolved_fact(fact):
                    unresolved_facts.append((round(candidate.confidence + surface_specificity * 0.18, 4), fact.model_copy(update={'canonical_entity_id': None, 'concept_entity_id': None, 'confidence': 0.0, 'matched_by': 'unresolved', 'resolution_kind': 'unresolved_candidate', 'resolved_entity_kind': None, 'knowledge_layer': None, 'target_eligible': False, 'unresolved_reason': 'rejected_by_business_lane_eligibility'})))
                    continue
                entity_key = fact.canonical_entity_id or fact.concept_entity_id or fact.candidate_text
                existing = resolved_facts.get(entity_key)
                if existing is None or fact.confidence > existing.confidence:
                    resolved_facts[entity_key] = fact
                continue
            unresolved_score = round(candidate.confidence + surface_specificity * 0.2, 4)
            unresolved_facts.append((unresolved_score, fact))
        if resolved_facts:
            return sorted(resolved_facts.values(), key=lambda item: (-item.confidence, item.entity_type or '', item.candidate_text))
        if unresolved_facts:
            unresolved_facts.sort(key=lambda item: (-item[0], item[1].candidate_text))
            return [fact for _, fact in unresolved_facts[:2]]
        return [self._unresolved_fact(mention, reason=self._no_candidate_reason(mention))]

    def annotate_batch_local_candidates(self, entity_facts, mentions):
        return cluster_unresolved_entity_facts(entity_facts, mentions)

    def _normalize_unresolved_reason(self, mention, candidate, raw_reason):
        candidate_text = candidate.normalized_text or candidate.text
        if self._matches_other_domain_catalog(candidate_text) or self._matches_other_domain_catalog(mention.normalized_text_compact):
            return 'blocked_by_other_domain_catalog'
        if raw_reason in {'ambiguous_alias'}:
            return 'ambiguous_multi_candidate'
        if raw_reason in {'noise_term', 'empty_candidate_text', 'insufficient_specificity_for_embedding', 'bad_weak_span'}:
            return 'bad_weak_span'
        if raw_reason == 'rejected_by_business_lane_eligibility':
            return 'rejected_by_business_lane_eligibility'
        if raw_reason == 'type_conflict':
            return 'type_conflict'
        if raw_reason in {'embedding_surface_support_insufficient', 'no_match_above_threshold'}:
            return 'candidate_found_but_low_confidence'
        return raw_reason or 'candidate_found_but_low_confidence'

    def _no_candidate_reason(self, mention):
        if self._matches_other_domain_catalog(mention.normalized_text_compact):
            return 'blocked_by_other_domain_catalog'
        normalized = mention.normalized_text_compact.strip()
        tokens = [token for token in normalized.split() if token]
        if not normalized or (len(tokens) <= 1 and len(normalized.replace(' ', '')) < 4):
            return 'bad_weak_span'
        return 'no_candidate_found'

    def _matches_other_domain_catalog(self, text):
        normalized = ' '.join(text.split()).casefold()
        if not normalized:
            return False
        compact = normalized.replace(' ', '')
        if normalized in self.other_domain_aliases or compact in self._other_domain_compact:
            return True
        return any((len(alias) >= 5 and alias in normalized for alias in self.other_domain_aliases))

    def _unresolved_fact(self, mention, *, reason):
        return EntityFact(mention_id=mention.mention_id, source_uap_id=mention.source_uap_id, candidate_text=mention.normalized_text_compact or mention.raw_text, entity_type=None, confidence=0.0, matched_by='unresolved', resolution_kind='unresolved_candidate', unresolved_reason=reason, canonical_candidate_ids=[], discovered_by=[], provenance=FactProvenance(source_uap_id=mention.source_uap_id, mention_id=mention.mention_id, provider_version=self.engine.embedding_provider.version, rule_version='entity-discovery-resolution-v4', evidence_text=mention.raw_text))

    def _keep_resolved_fact(self, fact):
        if fact.canonical_entity_id is not None:
            return True
        if fact.concept_entity_id is None:
            return False
        if not fact.target_eligible:
            return False
        if fact.matched_by in {'exact_alias', 'normalized_alias', 'compact_alias'} and fact.surface_specificity >= 0.3:
            return True
        if fact.matched_by == 'contextual_alias' and fact.knowledge_layer in {'domain', 'domain_overlay', 'review_overlay'}:
            return fact.confidence >= 0.9 and fact.surface_specificity >= 0.38
        return fact.confidence >= 0.94 and fact.surface_specificity >= 0.46
