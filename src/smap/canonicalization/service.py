from __future__ import annotations
import unicodedata
from dataclasses import dataclass, field
from rapidfuzz import fuzz
from smap.canonicalization.alias import AliasRegistry, boundary_contains, normalize_alias
from smap.canonicalization.models import CanonicalEntity, EntityAlias, EntityCandidate, ResolutionDecision, ResolutionMethod
from smap.ml.heads import BinaryLinearHead
from smap.ontology.prototypes import PrototypeRegistry
from smap.providers.base import EmbeddingProvider, EmbeddingPurpose, SimilarityMatch, VectorIndex, VectorItem, VectorNamespaceExpectation, VectorReuseState, VectorSearchHit
from smap.providers.fallback import TokenOverlapEmbeddingProvider
from smap.providers.vector_index_manifest import corpus_hash
MIN_CONTEXTUAL_TOKEN_OVERLAP = 0.6
MIN_EMBEDDING_DISAMBIGUATION_MARGIN = 0.06
_LOW_PRECISION_DISCOVERY = {'context_span', 'repeated_phrase', 'title_span', 'token_ngram'}
_HIGH_SIGNAL_DISCOVERY = {'alias_scan', 'brand_context', 'handle_pattern', 'hashtag', 'phobert_ner', 'product_code'}
_GENERIC_SURFACE_TOKENS = {'branch', 'brand', 'company', 'cua hang', 'factory', 'facility', 'hang', 'job', 'location', 'market', 'item', 'model', 'opening', 'plant', 'product', 'service', 'site', 'shop', 'store'}
_UNIQUE_PREFIX_ENTITY_TYPES = {'brand', 'organization', 'retailer', 'campaign', 'influencer'}
_TYPE_SPECIFICITY_FLOORS = {'brand': 0.08, 'organization': 0.12, 'retailer': 0.12, 'product': 0.08, 'product_line': 0.1, 'facility': 0.18, 'location': 0.18, 'person': 0.2, 'concept': 0.32}
_TYPE_EMBEDDING_FLOORS = {'brand': 0.1, 'organization': 0.14, 'retailer': 0.14, 'product': 0.1, 'product_line': 0.14, 'facility': 0.22, 'location': 0.22, 'person': 0.26, 'concept': 0.38}
_HIGH_SIGNAL_SOURCE_BONUS = {'alias_scan': 0.16, 'product_code': 0.18, 'brand_context': 0.12, 'phobert_ner': 0.12, 'hashtag': 0.08, 'title_span': 0.06}

def _significant_tokens(text):
    return {token for token in normalize_alias(text).split() if len(token) >= 3 or any((character.isdigit() for character in token))}

@dataclass(slots=True)
class _CandidateSignals:
    canonical_entity_id: str
    matched_alias: str | None = None
    exact_alias: float = 0.0
    normalized_alias: float = 0.0
    compact_alias: float = 0.0
    boundary_support: float = 0.0
    contextual_support: float = 0.0
    unique_prefix_support: float = 0.0
    fuzzy_support: float = 0.0
    overlay_support: float = 0.0
    signal_methods: set[str] = field(default_factory=set)

    def primary_method(self):
        if self.exact_alias >= 0.99:
            return 'exact_alias'
        if self.normalized_alias >= 0.92:
            return 'normalized_alias'
        if self.compact_alias >= 0.9:
            return 'compact_alias'
        if self.boundary_support >= max(self.contextual_support, self.unique_prefix_support, 0.5):
            return 'boundary_alias'
        if max(self.contextual_support, self.unique_prefix_support) >= 0.58:
            return 'contextual_alias'
        if self.fuzzy_support >= 0.93:
            return 'fuzzy_alias'
        return 'embedding_similarity_ranked'

class CanonicalizationEngine:

    def __init__(self, alias_registry, embedding_provider=None, prototype_registry=None, vector_index=None, mention_worthiness_head=None, entity_link_head=None, fuzzy_threshold=91.0, embedding_threshold=0.75, vector_namespace='canonical_entities', embedding_rerank_enabled=True):
        self.alias_registry = alias_registry
        self.embedding_provider = embedding_provider or TokenOverlapEmbeddingProvider()
        self.prototype_registry = prototype_registry
        self.vector_index = vector_index
        self.mention_worthiness_head = mention_worthiness_head
        self.entity_link_head = entity_link_head
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold
        self.vector_namespace = vector_namespace
        self.embedding_rerank_enabled = embedding_rerank_enabled
        self._all_aliases = tuple(self.alias_registry.all_aliases())
        self._alias_token_index = self._build_alias_token_index(self._all_aliases)
        self._compact_alias_index = self._build_compact_alias_index(self._all_aliases)
        self._entity_strings = {entity_id: bundle.alias_lookup_text for entity_id, bundle in self.prototype_registry.entities.items()} if self.prototype_registry is not None else self.alias_registry.candidate_strings()
        self._vector_alias_strings = self.alias_registry.vector_alias_strings()
        self._candidate_index_ready = False
        self._embedding_rank_cache: dict[tuple[str, tuple[str, ...]], list[SimilarityMatch]] = {}
        if self.vector_index is not None and self._vector_alias_strings:
            self._ensure_vector_index()

    def resolve(self, candidate):
        normalized_candidate = candidate.normalized_text or normalize_alias(candidate.text)
        if not normalized_candidate:
            return self._unresolved(candidate, reason='empty_candidate_text')
        if self.alias_registry.is_noise_term(normalized_candidate):
            return self._unresolved(candidate, reason='noise_term')
        signal_map = self._collect_candidate_signals(candidate, normalized_candidate)
        if not self._candidate_is_mention_worthy(candidate, normalized_candidate, signal_map=signal_map):
            return self._unresolved(candidate, reason='bad_weak_span')
        if not signal_map and (not self._should_attempt_embedding(candidate, normalized_candidate)):
            return self._unresolved(candidate, reason='bad_weak_span')
        shortlist = self._candidate_shortlist_ids(candidate, normalized_candidate, signal_map=signal_map)
        if not shortlist:
            return self._unresolved(candidate, reason='no_candidate_found')
        ranked_matches = self._rank_embedding_candidates(candidate, normalized_candidate, shortlist=shortlist, signal_map=signal_map)
        top_match = ranked_matches[0] if ranked_matches else None
        if top_match is None:
            return self._unresolved(candidate, reason='no_match_above_threshold')
        if candidate.entity_type_hint is not None and top_match.metadata.get('type_consistent') is False and (top_match.score >= max(self.embedding_threshold - 0.05, 0.6)):
            return self._unresolved(candidate, reason='type_conflict', candidate_canonical_ids=[match.candidate_id for match in ranked_matches])
        if self._embedding_result_is_ambiguous(candidate, ranked_matches):
            return self._unresolved(candidate, reason='ambiguous_multi_candidate', candidate_canonical_ids=[match.candidate_id for match in ranked_matches])
        score_floor = self._candidate_score_floor(candidate, signal_map.get(top_match.candidate_id))
        if top_match.score < score_floor:
            unresolved_reason = 'embedding_surface_support_insufficient' if _metadata_float(top_match.metadata.get('alias_surface_support')) >= 0.38 else 'no_match_above_threshold'
            return self._unresolved(candidate, reason=unresolved_reason, candidate_canonical_ids=[match.candidate_id for match in ranked_matches])
        if not self._embedding_surface_support(candidate, top_match.candidate_id, signal=signal_map.get(top_match.candidate_id)):
            return self._unresolved(candidate, reason='embedding_surface_support_insufficient', candidate_canonical_ids=[match.candidate_id for match in ranked_matches])
        return self._resolve_ranked_match(candidate, ranked_matches, signal_map=signal_map)

    def _collect_candidate_signals(self, candidate, normalized_candidate):
        signal_map: dict[str, _CandidateSignals] = {}
        for alias in self.alias_registry.find_exact_aliases(candidate.text):
            self._update_alias_signal(signal_map, alias, signal_kind='exact_alias', score=1.0)
        for alias in self.alias_registry.find_normalized_aliases(normalized_candidate):
            self._update_alias_signal(signal_map, alias, signal_kind='normalized_alias', score=0.97)
        compact = normalized_candidate.replace(' ', '')
        if len(compact) >= 3:
            for alias in self._compact_alias_index.get(compact, []):
                self._update_alias_signal(signal_map, alias, signal_kind='compact_alias', score=0.94)
        normalized_context = normalize_alias(candidate.context_text or '')
        candidate_tokens = _significant_tokens(normalized_candidate)
        for alias in self._prefilter_aliases(normalized_candidate):
            alias_tokens = _significant_tokens(alias.normalized_alias)
            if alias.normalized_alias != normalized_candidate and boundary_contains(normalized_candidate, alias.normalized_alias):
                coverage = len(alias.normalized_alias) / max(len(normalized_candidate), 1)
                if candidate.entity_type_hint is not None:
                    entity = self.alias_registry.entities[alias.canonical_entity_id]
                    if entity.entity_type == candidate.entity_type_hint:
                        coverage += 0.05
                self._update_alias_signal(signal_map, alias, signal_kind='boundary_alias', score=min(coverage, 0.96))
            if normalized_context and alias_tokens and candidate_tokens:
                overlap = len(candidate_tokens & alias_tokens) / len(candidate_tokens)
                if overlap >= MIN_CONTEXTUAL_TOKEN_OVERLAP and boundary_contains(normalized_context, alias.normalized_alias):
                    self._update_alias_signal(signal_map, alias, signal_kind='contextual_alias', score=min(0.58 + overlap * 0.22, 0.9))
            if self._looks_like_unique_prefix_candidate(candidate, normalized_candidate, alias):
                prefix_score = 0.72 + (0.04 if candidate.entity_type_hint == self.alias_registry.entities[alias.canonical_entity_id].entity_type else 0.0)
                self._update_alias_signal(signal_map, alias, signal_kind='unique_prefix', score=min(prefix_score, 0.86))
            if len(normalized_candidate) >= 4 and abs(len(alias.normalized_alias) - len(normalized_candidate)) <= 12:
                fuzzy_score = max(float(fuzz.ratio(normalized_candidate, alias.normalized_alias)) / 100.0, float(fuzz.token_set_ratio(normalized_candidate, alias.normalized_alias)) / 100.0)
                if fuzzy_score >= 0.91:
                    self._update_alias_signal(signal_map, alias, signal_kind='fuzzy_alias', score=min(fuzzy_score, 0.9))
        return signal_map

    def _update_alias_signal(self, signal_map, alias, *, signal_kind, score):
        signal = signal_map.setdefault(alias.canonical_entity_id, _CandidateSignals(canonical_entity_id=alias.canonical_entity_id))
        signal.matched_alias = signal.matched_alias or alias.alias
        if signal_kind == 'exact_alias':
            signal.exact_alias = max(signal.exact_alias, score)
        elif signal_kind == 'normalized_alias':
            signal.normalized_alias = max(signal.normalized_alias, score)
        elif signal_kind == 'compact_alias':
            signal.compact_alias = max(signal.compact_alias, score)
        elif signal_kind == 'boundary_alias':
            signal.boundary_support = max(signal.boundary_support, score)
        elif signal_kind == 'contextual_alias':
            signal.contextual_support = max(signal.contextual_support, score)
        elif signal_kind == 'unique_prefix':
            signal.unique_prefix_support = max(signal.unique_prefix_support, score)
        elif signal_kind == 'fuzzy_alias':
            signal.fuzzy_support = max(signal.fuzzy_support, score)
        if alias.source != 'ontology':
            signal.overlay_support = max(signal.overlay_support, 0.06)
        signal.signal_methods.add(signal_kind)

    def _candidate_shortlist_ids(self, candidate, normalized_candidate, *, signal_map):
        shortlisted = list(signal_map)
        prototype_shortlist = self._shortlist_prototype_candidates(candidate, normalized_candidate, seed_candidate_ids=set(signal_map))
        for entity_id in prototype_shortlist:
            if entity_id not in shortlisted:
                shortlisted.append(entity_id)
        return shortlisted

    def _looks_like_unique_prefix_candidate(self, candidate, normalized_candidate, alias):
        candidate_tokens = _significant_tokens(normalized_candidate)
        if len(candidate_tokens) != 1:
            return False
        token = next(iter(candidate_tokens))
        if len(token) < 4 or token in _GENERIC_SURFACE_TOKENS:
            return False
        entity = self.alias_registry.entities[alias.canonical_entity_id]
        if entity.entity_type not in _UNIQUE_PREFIX_ENTITY_TYPES:
            return False
        if candidate.entity_type_hint is not None and entity.entity_type != candidate.entity_type_hint:
            return False
        alias_tokens = alias.normalized_alias.split()
        compact_alias = alias.normalized_alias.replace(' ', '')
        if not (alias_tokens and alias_tokens[0] == token or compact_alias.startswith(token)):
            return False
        specificity = self._surface_specificity(candidate, normalized_candidate)
        return specificity >= 0.1 or self._looks_distinctive_surface(candidate.text)

    def _candidate_score_floor(self, candidate, signal):
        floor = self.embedding_threshold
        if signal is None:
            return floor
        if signal.exact_alias >= 0.99:
            return 0.78
        if signal.normalized_alias >= 0.95 or signal.compact_alias >= 0.9:
            return 0.72
        if max(signal.boundary_support, signal.contextual_support, signal.unique_prefix_support) >= 0.66:
            return max(self.embedding_threshold - 0.08, 0.66)
        if signal.fuzzy_support >= 0.93:
            return max(self.embedding_threshold, 0.8)
        if candidate.entity_type_hint == 'concept':
            return max(self.embedding_threshold + 0.04, 0.8)
        return floor

    def _resolve_ranked_match(self, candidate, ranked_matches, *, signal_map):
        top_match = ranked_matches[0]
        entity = self.alias_registry.entities[top_match.candidate_id]
        signal = signal_map.get(top_match.candidate_id)
        if signal is None:
            matched_by: ResolutionMethod = 'embedding_similarity_ranked' if len(ranked_matches) > 1 else 'embedding_similarity'
            matched_alias = entity.name
        else:
            matched_by = signal.primary_method()
            matched_alias = signal.matched_alias or entity.name
        provider_suffix = 'candidate-fusion-v5'
        return self._decision_for_entity(candidate, entity, matched_alias=matched_alias, matched_by=matched_by, confidence=round(top_match.score, 3), provider_version=f'{self.embedding_provider.version}+{provider_suffix}', candidate_canonical_ids=[match.candidate_id for match in ranked_matches])

    def _embedding_query_text(self, candidate):
        parts = [candidate.text, candidate.context_text or '', candidate.surrounding_text or '']
        return ' || '.join((part for part in parts if part))

    def _embedding_view_texts(self, candidate):
        span_text = candidate.normalized_text or normalize_alias(candidate.text)
        local_text = normalize_alias(candidate.surrounding_text or candidate.text)
        full_text = normalize_alias(candidate.full_text or candidate.surrounding_text or candidate.text)
        thread_text = normalize_alias(candidate.context_text or candidate.full_text or candidate.text)
        return (span_text, local_text, full_text, thread_text)

    def _embedding_surface_support(self, candidate, candidate_id, *, signal=None):
        normalized_candidate = candidate.normalized_text or normalize_alias(candidate.text)
        candidate_tokens = _significant_tokens(normalized_candidate)
        if not candidate_tokens:
            return False
        if signal is not None:
            if signal.exact_alias >= 0.99 or signal.normalized_alias >= 0.95 or signal.compact_alias >= 0.9:
                return True
            if max(signal.boundary_support, signal.contextual_support, signal.unique_prefix_support) >= 0.62:
                return True
        specificity = self._surface_specificity(candidate, normalized_candidate)
        if specificity < 0.15 and len(candidate_tokens) == 1 and (normalized_candidate in _GENERIC_SURFACE_TOKENS):
            return False
        if any((method in _HIGH_SIGNAL_DISCOVERY for method in candidate.discovered_by)):
            return True
        if any((character.isdigit() for character in normalized_candidate)):
            return True
        type_floor = _TYPE_SPECIFICITY_FLOORS.get(candidate.entity_type_hint or '', 0.18)
        if specificity < type_floor:
            return False
        alias_support = self._alias_surface_support(normalized_candidate, candidate_id)
        if candidate.discovered_by and all((method in _LOW_PRECISION_DISCOVERY for method in candidate.discovered_by)):
            return alias_support >= max(0.48, 0.62 - specificity * 0.08)
        if alias_support >= 0.48 and specificity >= 0.3:
            return True
        if len(candidate_tokens) >= 2 and alias_support >= 0.3 and (specificity >= 0.45):
            return True
        return alias_support >= max(0.36, 0.58 - specificity * 0.1)

    def _candidate_is_mention_worthy(self, candidate, normalized_candidate, *, signal_map):
        specificity = self._surface_specificity(candidate, normalized_candidate)
        signal_map = signal_map or {}
        if any((signal.exact_alias >= 0.99 or signal.normalized_alias >= 0.92 or signal.compact_alias >= 0.9 or (max(signal.boundary_support, signal.contextual_support, signal.unique_prefix_support) >= 0.62) for signal in signal_map.values())):
            return True
        if any((method in _HIGH_SIGNAL_DISCOVERY for method in candidate.discovered_by)):
            return True
        if any((character.isdigit() for character in normalized_candidate)):
            return True
        token_count = len(_significant_tokens(normalized_candidate))
        type_floor = _TYPE_SPECIFICITY_FLOORS.get(candidate.entity_type_hint or '', 0.16)
        if token_count >= 2 and specificity >= type_floor:
            return True
        if candidate.entity_type_hint in {'brand', 'organization', 'retailer'} and self._looks_distinctive_surface(candidate.text):
            return specificity >= max(type_floor - 0.06, 0.06)
        prototype_top_score = 0.0
        if self.prototype_registry is None:
            if self.mention_worthiness_head is not None:
                mention_score = self.mention_worthiness_head.score(self._mention_worthiness_features(candidate, normalized_candidate, signal_map=signal_map, specificity=specificity, prototype_top_score=prototype_top_score))
                return mention_score >= 0.48
            return specificity >= max(type_floor, 0.25)
        shortlisted = self._shortlist_prototype_candidates(candidate, normalized_candidate, seed_candidate_ids=set(signal_map))
        if not shortlisted:
            return False
        ranked = self._rank_prototype_candidates(candidate, normalized_candidate, shortlisted, signal_map=signal_map)
        prototype_top_score = ranked[0].score if ranked else 0.0
        if self.mention_worthiness_head is not None:
            mention_score = self.mention_worthiness_head.score(self._mention_worthiness_features(candidate, normalized_candidate, signal_map=signal_map, specificity=specificity, prototype_top_score=prototype_top_score))
            threshold = 0.46 if candidate.entity_type_hint in {'brand', 'product', 'product_line'} else 0.5
            return mention_score >= threshold
        required_score = 0.54 if token_count <= 1 else 0.5
        if candidate.entity_type_hint is None and candidate.discovered_by and all((method in _LOW_PRECISION_DISCOVERY for method in candidate.discovered_by)):
            required_score += 0.06
        return bool(ranked and ranked[0].score >= required_score)

    def _should_attempt_embedding(self, candidate, normalized_candidate):
        specificity = self._surface_specificity(candidate, normalized_candidate)
        candidate_tokens = _significant_tokens(normalized_candidate)
        if specificity < 0.15 and len(candidate_tokens) == 1 and (normalized_candidate in _GENERIC_SURFACE_TOKENS):
            return False
        if any((method in _HIGH_SIGNAL_DISCOVERY for method in candidate.discovered_by)):
            return True
        if any((character.isdigit() for character in normalized_candidate)):
            return True
        type_floor = _TYPE_EMBEDDING_FLOORS.get(candidate.entity_type_hint or '', 0.22)
        if len(candidate_tokens) >= 2 and specificity >= type_floor:
            return True
        if candidate.entity_type_hint is None and candidate.discovered_by and all((method in _LOW_PRECISION_DISCOVERY for method in candidate.discovered_by)):
            return specificity >= max(type_floor + 0.18, 0.5)
        if candidate.context_text and len(candidate.context_text.split()) >= 4 and (specificity >= max(type_floor + 0.08, 0.35)):
            return True
        return specificity >= max(type_floor + 0.12, 0.45)

    def _shortlist_prototype_candidates(self, candidate, normalized_candidate, *, seed_candidate_ids=None):
        if self.prototype_registry is None:
            return []
        prefetched = list(seed_candidate_ids or ())
        prefetched.extend({alias.canonical_entity_id for alias in self._prefilter_aliases(normalized_candidate)})
        if candidate.entity_type_hint is not None:
            type_filtered = [entity_id for entity_id, bundle in self.prototype_registry.entities.items() if bundle.entity_type == candidate.entity_type_hint]
            prefetched.extend(type_filtered[:24])
        if not prefetched:
            prefetched.extend(self.prototype_registry.entities.keys())
        seen: set[str] = set()
        shortlisted: list[str] = []
        for entity_id in prefetched:
            if entity_id in seen:
                continue
            seen.add(entity_id)
            shortlisted.append(entity_id)
            if len(shortlisted) >= 24:
                break
        return shortlisted

    def _rank_embedding_candidates(self, candidate, normalized_candidate, *, shortlist=None, signal_map=None):
        if not self._entity_strings or not self.embedding_rerank_enabled:
            return []
        if self.prototype_registry is None and signal_map:
            shortlist = shortlist or list(signal_map)
            ranked = self._rank_signal_candidates(candidate, normalized_candidate, shortlist, signal_map=signal_map)
            if ranked:
                return ranked
        if self.prototype_registry is None:
            return self._rank_candidate_pool(self._embedding_query_text(candidate), self._entity_strings)
        shortlist = shortlist or self._shortlist_prototype_candidates(candidate, normalized_candidate)
        if not shortlist:
            return []
        return self._rank_prototype_candidates(candidate, normalized_candidate, shortlist, signal_map=signal_map or {})

    def _rank_signal_candidates(self, candidate, normalized_candidate, candidate_ids, *, signal_map):
        specificity = self._surface_specificity(candidate, normalized_candidate)
        compact_candidate = normalized_candidate.replace(' ', '')
        normalized_context = normalize_alias(' || '.join((part for part in (candidate.context_text, candidate.full_text) if part)))
        ranked: list[SimilarityMatch] = []
        for candidate_id in candidate_ids:
            entity = self.alias_registry.entities.get(candidate_id)
            signal = signal_map.get(candidate_id)
            if entity is None or signal is None:
                continue
            alias_surface_support = self._alias_surface_support(normalized_candidate, candidate_id)
            type_consistent = candidate.entity_type_hint is None or candidate.entity_type_hint == entity.entity_type
            type_bonus = 0.0
            if candidate.entity_type_hint is not None:
                type_bonus = 0.14 if type_consistent else -0.16
            compact_exact = max(1.0 if compact_candidate and compact_candidate in self._entity_compact_aliases(candidate_id) else 0.0, signal.compact_alias)
            source_bonus = max((_HIGH_SIGNAL_SOURCE_BONUS.get(method, 0.0) for method in candidate.discovered_by), default=0.0)
            generic_penalty = 0.08 if specificity < 0.15 and entity.entity_kind == 'entity' else 0.0
            concept_penalty = 0.18 if entity.entity_kind == 'concept' and specificity >= 0.34 and (compact_exact <= 0.0) else 0.0
            anti_confusion_penalty = 0.34 if entity.anti_confusion_phrases and normalized_context and any((phrase in normalized_context for phrase in entity.anti_confusion_phrases[:6])) and (specificity < 0.32) else 0.0
            signal_confidence = max(0.0, 0.94 + signal.exact_alias * 0.04 if signal.exact_alias > 0.0 else 0.0, 0.88 + signal.normalized_alias * 0.08 if signal.normalized_alias > 0.0 else 0.0, 0.84 + compact_exact * 0.12 if compact_exact > 0.0 else 0.0, 0.52 + signal.boundary_support * 0.38 if signal.boundary_support > 0.0 else 0.0, 0.5 + signal.contextual_support * 0.34 if signal.contextual_support > 0.0 else 0.0, 0.5 + signal.unique_prefix_support * 0.3 if signal.unique_prefix_support > 0.0 else 0.0, 0.42 + signal.fuzzy_support * 0.24 if signal.fuzzy_support > 0.0 else 0.0)
            score = signal_confidence + alias_surface_support * 0.08 + min(candidate.confidence, 1.0) * 0.04 + source_bonus + signal.overlay_support + type_bonus - generic_penalty - concept_penalty - anti_confusion_penalty
            features = self._entity_link_features(alias_exact=signal.exact_alias, normalized_alias=signal.normalized_alias, compact_alias=compact_exact, boundary_support=signal.boundary_support, contextual_alias_support=signal.contextual_support, unique_prefix_support=signal.unique_prefix_support, fuzzy_support=signal.fuzzy_support, alias_surface_support=alias_surface_support, span_similarity=0.0, local_similarity=0.0, full_similarity=0.0, thread_similarity=0.0, candidate_confidence=min(candidate.confidence, 1.0), source_bonus=source_bonus, contextual_support=0.0, overlay_support=signal.overlay_support, layer_bonus=0.04 if entity.knowledge_layer != 'base' else 0.0, entity_kind_bonus=0.03 if entity.entity_kind == 'entity' and entity.target_eligible else 0.0, type_bonus=type_bonus, concept_penalty=concept_penalty, generic_penalty=generic_penalty, anti_confusion_penalty=anti_confusion_penalty, specificity=specificity, type_consistent=type_consistent, entity_kind=entity.entity_kind, entity_type=entity.entity_type, target_eligible=entity.target_eligible)
            learned_score = self.entity_link_head.score(features) if self.entity_link_head is not None else None
            final_score = learned_score * 0.82 + score * 0.18 if learned_score is not None else score
            ranked.append(SimilarityMatch(candidate_id=candidate_id, score=round(max(min(final_score, 0.995), 0.0), 6), candidate_text=entity.name, metadata={**features, 'alias_surface_support': round(alias_surface_support, 6), 'contextual_support': round(signal.contextual_support, 6), 'boundary_support': round(signal.boundary_support, 6), 'unique_prefix_support': round(signal.unique_prefix_support, 6), 'fuzzy_support': round(signal.fuzzy_support, 6), 'normalized_alias': round(signal.normalized_alias, 6), 'compact_alias': round(compact_exact, 6), 'type_consistent': type_consistent, 'candidate_entity_type': entity.entity_type, 'candidate_entity_kind': entity.entity_kind, 'anti_confusion_penalty': round(anti_confusion_penalty, 6), 'heuristic_score': round(score, 6), 'learned_score': round(learned_score, 6) if learned_score is not None else None}))
        return sorted(ranked, key=lambda item: (-item.score, item.candidate_id))[:5]

    def _rank_prototype_candidates(self, candidate, normalized_candidate, candidate_ids, *, signal_map):
        if self.prototype_registry is None:
            return []
        bundles = [self.prototype_registry.entities[candidate_id] for candidate_id in candidate_ids if candidate_id in self.prototype_registry.entities and self.prototype_registry.entities[candidate_id].vector is not None]
        if not bundles:
            return []
        span_text, local_text, full_text, thread_text = self._embedding_view_texts(candidate)
        span_vector, local_vector, full_vector, thread_vector = self.embedding_provider.embed_texts([span_text, local_text, full_text, thread_text], purpose=EmbeddingPurpose.LINKING)
        compact_candidate = normalized_candidate.replace(' ', '')
        normalized_context = normalize_alias(' || '.join((part for part in (candidate.context_text, candidate.full_text) if part)))
        specificity = self._surface_specificity(candidate, normalized_candidate)
        ranked: list[SimilarityMatch] = []
        for bundle in bundles:
            if bundle.vector is None:
                continue
            signal = signal_map.get(bundle.canonical_entity_id)
            type_consistent = candidate.entity_type_hint is None or candidate.entity_type_hint == bundle.entity_type
            alias_exact = max(1.0 if normalized_candidate in bundle.normalized_aliases else 0.0, signal.exact_alias if signal else 0.0)
            normalized_alias = signal.normalized_alias if signal is not None else 0.0
            compact_exact = max(1.0 if compact_candidate and compact_candidate in bundle.compact_aliases else 0.0, signal.compact_alias if signal else 0.0)
            alias_surface_support = self._alias_surface_support(normalized_candidate, bundle.canonical_entity_id)
            span_similarity = max(sum((a * b for a, b in zip(span_vector, bundle.vector, strict=True))), 0.0)
            local_similarity = max(sum((a * b for a, b in zip(local_vector, bundle.vector, strict=True))), 0.0)
            full_similarity = max(sum((a * b for a, b in zip(full_vector, bundle.vector, strict=True))), 0.0)
            thread_similarity = max(sum((a * b for a, b in zip(thread_vector, bundle.vector, strict=True))), 0.0)
            boundary_support = signal.boundary_support if signal is not None else 0.0
            contextual_alias_support = signal.contextual_support if signal is not None else 0.0
            unique_prefix_support = signal.unique_prefix_support if signal is not None else 0.0
            fuzzy_support = signal.fuzzy_support if signal is not None else 0.0
            type_bonus = 0.0
            if candidate.entity_type_hint is not None:
                type_bonus = 0.14 if candidate.entity_type_hint == bundle.entity_type else -0.16
            source_bonus = max((_HIGH_SIGNAL_SOURCE_BONUS.get(method, 0.0) for method in candidate.discovered_by), default=0.0)
            contextual_support = self._prototype_context_support(bundle, normalized_context)
            anti_confusion_penalty = 0.32 if bundle.anti_confusion_phrases and normalized_context and any((phrase in normalized_context for phrase in bundle.anti_confusion_phrases[:6])) and (specificity < 0.32) else 0.0
            layer_bonus = 0.04 if bundle.knowledge_layer != 'base' and (alias_surface_support >= 0.4 or contextual_support > 0.0 or (signal is not None and signal.overlay_support > 0.0)) else 0.0
            entity_kind_bonus = 0.03 if bundle.entity_kind == 'entity' and bundle.target_eligible else 0.0
            concept_penalty = 0.18 if bundle.entity_kind == 'concept' and specificity >= 0.34 and (not alias_exact) and (not compact_exact) else 0.0
            generic_penalty = 0.08 if specificity < 0.15 and bundle.entity_kind == 'entity' else 0.0
            score = alias_exact * 0.29 + normalized_alias * 0.16 + compact_exact * 0.22 + boundary_support * 0.12 + contextual_alias_support * 0.1 + unique_prefix_support * 0.06 + fuzzy_support * 0.04 + alias_surface_support * 0.14 + span_similarity * 0.11 + local_similarity * 0.1 + full_similarity * 0.07 + thread_similarity * 0.04 + min(candidate.confidence, 1.0) * 0.05 + source_bonus + contextual_support + (signal.overlay_support if signal is not None else 0.0) + layer_bonus + entity_kind_bonus + type_bonus - concept_penalty - generic_penalty - anti_confusion_penalty
            features = self._entity_link_features(alias_exact=alias_exact, normalized_alias=normalized_alias, compact_alias=compact_exact, boundary_support=boundary_support, contextual_alias_support=contextual_alias_support, unique_prefix_support=unique_prefix_support, fuzzy_support=fuzzy_support, alias_surface_support=alias_surface_support, span_similarity=span_similarity, local_similarity=local_similarity, full_similarity=full_similarity, thread_similarity=thread_similarity, candidate_confidence=min(candidate.confidence, 1.0), source_bonus=source_bonus, contextual_support=contextual_support, overlay_support=signal.overlay_support if signal is not None else 0.0, layer_bonus=layer_bonus, entity_kind_bonus=entity_kind_bonus, type_bonus=type_bonus, concept_penalty=concept_penalty, generic_penalty=generic_penalty, anti_confusion_penalty=anti_confusion_penalty, specificity=specificity, type_consistent=type_consistent, entity_kind=bundle.entity_kind, entity_type=bundle.entity_type, target_eligible=bundle.target_eligible)
            learned_score = self.entity_link_head.score(features) if self.entity_link_head is not None else None
            final_score = learned_score * 0.84 + score * 0.16 if learned_score is not None else score
            ranked.append(SimilarityMatch(candidate_id=bundle.canonical_entity_id, score=round(max(min(final_score, 0.995), 0.0), 6), candidate_text=bundle.label, metadata={**features, 'span_similarity': round(span_similarity, 6), 'local_similarity': round(local_similarity, 6), 'full_similarity': round(full_similarity, 6), 'thread_similarity': round(thread_similarity, 6), 'normalized_alias': round(normalized_alias, 6), 'boundary_support': round(boundary_support, 6), 'contextual_alias_support': round(contextual_alias_support, 6), 'unique_prefix_support': round(unique_prefix_support, 6), 'fuzzy_support': round(fuzzy_support, 6), 'alias_surface_support': round(alias_surface_support, 6), 'source_bonus': round(source_bonus, 6), 'contextual_support': round(contextual_support, 6), 'layer_bonus': round(layer_bonus, 6), 'anti_confusion_penalty': round(anti_confusion_penalty, 6), 'candidate_entity_type': bundle.entity_type, 'candidate_entity_kind': bundle.entity_kind, 'type_consistent': type_consistent, 'heuristic_score': round(score, 6), 'learned_score': round(learned_score, 6) if learned_score is not None else None}))
        return sorted(ranked, key=lambda item: (-item.score, item.candidate_id))[:5]

    def _mention_worthiness_features(self, candidate, normalized_candidate, *, signal_map, specificity, prototype_top_score):
        signals = list((signal_map or {}).values())
        token_count = len(_significant_tokens(normalized_candidate))
        strongest_alias = max((max(signal.exact_alias, signal.normalized_alias, signal.compact_alias, signal.boundary_support, signal.contextual_support, signal.unique_prefix_support, signal.fuzzy_support) for signal in signals), default=0.0)
        return {'specificity': specificity, 'token_count': min(float(token_count) / 4.0, 1.0), 'has_digits': 1.0 if any((character.isdigit() for character in normalized_candidate)) else 0.0, 'candidate_confidence': min(candidate.confidence, 1.0), 'strongest_alias_signal': strongest_alias, 'prototype_top_score': prototype_top_score, 'is_high_signal_discovery': 1.0 if any((method in _HIGH_SIGNAL_DISCOVERY for method in candidate.discovered_by)) else 0.0, 'is_low_precision_only': 1.0 if candidate.discovered_by and all((method in _LOW_PRECISION_DISCOVERY for method in candidate.discovered_by)) else 0.0, 'type_brand_like': 1.0 if candidate.entity_type_hint in {'brand', 'organization', 'retailer'} else 0.0, 'type_product_like': 1.0 if candidate.entity_type_hint in {'product', 'product_line'} else 0.0, 'type_concept': 1.0 if candidate.entity_type_hint == 'concept' else 0.0, 'distinctive_surface': 1.0 if self._looks_distinctive_surface(candidate.text) else 0.0}

    def _entity_link_features(self, *, alias_exact, normalized_alias, compact_alias, boundary_support, contextual_alias_support, unique_prefix_support, fuzzy_support, alias_surface_support, span_similarity, local_similarity, full_similarity, thread_similarity, candidate_confidence, source_bonus, contextual_support, overlay_support, layer_bonus, entity_kind_bonus, type_bonus, concept_penalty, generic_penalty, anti_confusion_penalty, specificity, type_consistent, entity_kind, entity_type, target_eligible):
        return {'alias_exact': alias_exact, 'normalized_alias': normalized_alias, 'compact_alias': compact_alias, 'boundary_support': boundary_support, 'contextual_alias_support': contextual_alias_support, 'unique_prefix_support': unique_prefix_support, 'fuzzy_support': fuzzy_support, 'alias_surface_support': alias_surface_support, 'span_similarity': span_similarity, 'local_similarity': local_similarity, 'full_similarity': full_similarity, 'thread_similarity': thread_similarity, 'candidate_confidence': candidate_confidence, 'source_bonus': source_bonus, 'contextual_support': contextual_support, 'overlay_support': overlay_support, 'layer_bonus': layer_bonus, 'entity_kind_bonus': entity_kind_bonus, 'type_bonus': type_bonus, 'concept_penalty': concept_penalty, 'generic_penalty': generic_penalty, 'anti_confusion_penalty': anti_confusion_penalty, 'specificity': specificity, 'type_consistent': 1.0 if type_consistent else 0.0, 'entity_is_concept': 1.0 if entity_kind == 'concept' else 0.0, 'entity_is_product': 1.0 if entity_type in {'product', 'product_line'} else 0.0, 'entity_is_brand_like': 1.0 if entity_type in {'brand', 'organization', 'retailer'} else 0.0, 'target_eligible': 1.0 if target_eligible else 0.0}

    def _surface_specificity(self, candidate, normalized_candidate):
        tokens = [token for token in normalized_candidate.split() if token]
        if not tokens:
            return 0.0
        if normalize_alias(candidate.text) in _GENERIC_SURFACE_TOKENS:
            return 0.0
        score = 0.0
        if any((character.isdigit() for character in normalized_candidate)):
            score += 0.35
        if len(tokens) >= 2:
            score += 0.25
        if max((len(token) for token in tokens), default=0) >= 6:
            score += 0.15
        if candidate.entity_type_hint in {'product', 'brand', 'retailer', 'organization'}:
            score += 0.15
        if any((method in _HIGH_SIGNAL_DISCOVERY for method in candidate.discovered_by)):
            score += 0.2
        if any((token in _GENERIC_SURFACE_TOKENS for token in tokens)):
            score -= 0.2
        return round(max(min(score, 1.0), 0.0), 4)

    def _looks_distinctive_surface(self, text):
        stripped = text.strip()
        if len(stripped) < 4:
            return False
        return any((character.isupper() for character in stripped)) or any((character.isdigit() for character in stripped))

    def _alias_surface_support(self, normalized_candidate, candidate_id):
        best_score = 0.0
        candidate_tokens = _significant_tokens(normalized_candidate)
        for alias in self._prefilter_aliases(normalized_candidate):
            if alias.canonical_entity_id != candidate_id:
                continue
            alias_tokens = _significant_tokens(alias.normalized_alias)
            overlap = 0.0
            if candidate_tokens and alias_tokens:
                overlap = len(candidate_tokens & alias_tokens) / len(candidate_tokens)
            fuzzy_score = max(float(fuzz.ratio(normalized_candidate, alias.normalized_alias)) / 100.0, float(fuzz.token_set_ratio(normalized_candidate, alias.normalized_alias)) / 100.0)
            boundary_bonus = 0.0
            if boundary_contains(normalized_candidate, alias.normalized_alias) or boundary_contains(alias.normalized_alias, normalized_candidate):
                boundary_bonus = 0.15
            best_score = max(best_score, overlap, min(fuzzy_score + boundary_bonus, 1.0))
        return round(best_score, 6)

    def _evidence_span(self, candidate):
        if candidate.start_char is None or candidate.end_char is None:
            return None
        return (candidate.start_char, candidate.end_char)

    def _ensure_vector_index(self):
        items = self._vector_items()
        if not items or self.vector_index is None:
            return
        expected = VectorNamespaceExpectation(namespace=self.vector_namespace, backend=self.vector_index.provenance.provider_name, dimension=len(items[0].vector), normalization_mode='cosine_normalized', embedding_model_id=self.embedding_provider.provenance.model_id, embedding_provider_name=self.embedding_provider.provenance.provider_name, embedding_provider_version=self.embedding_provider.provenance.provider_version, embedding_purpose=EmbeddingPurpose.PASSAGE.value, corpus_hash=corpus_hash(items))
        self.vector_index.bind_expectation(namespace=self.vector_namespace, expected=expected)
        info = self.vector_index.info(namespace=self.vector_namespace, expected=expected)
        if info is not None and info.reuse_state == VectorReuseState.VALID.value and self.vector_index.load(namespace=self.vector_namespace, expected=expected):
            self._candidate_index_ready = True
            return
        self._build_vector_index(items)

    def _vector_items(self):
        if self.vector_index is None or not self._vector_alias_strings:
            return []
        alias_items = list(self._vector_alias_strings.items())
        vectors = self.embedding_provider.embed_texts([alias_text for _, (_, alias_text, _) in alias_items], purpose=EmbeddingPurpose.PASSAGE)
        return [VectorItem(item_id=alias_id, vector=vector, text=alias_text, metadata={'canonical_entity_id': canonical_entity_id, 'alias_text': alias_text, 'alias_source': alias_source, 'embedding_model_id': self.embedding_provider.provenance.model_id, 'embedding_provider_name': self.embedding_provider.provenance.provider_name, 'embedding_provider_version': self.embedding_provider.provenance.provider_version, 'embedding_purpose': EmbeddingPurpose.PASSAGE.value}) for (alias_id, (canonical_entity_id, alias_text, alias_source)), vector in zip(alias_items, vectors, strict=True)]

    def _build_vector_index(self, items=None):
        if self.vector_index is None or not self._vector_alias_strings:
            return
        items = items or self._vector_items()
        self.vector_index.reset(namespace=self.vector_namespace)
        self.vector_index.upsert(items, namespace=self.vector_namespace)
        self._candidate_index_ready = True

    def vector_items(self):
        return self._vector_items()

    def _rank_candidate_pool(self, text, candidate_texts):
        cache_key = (text, tuple(sorted(candidate_texts)))
        cached = self._embedding_rank_cache.get(cache_key)
        if cached is not None:
            return cached
        query_variants = _query_variants(text)
        aggregated: dict[str, SimilarityMatch] = {}
        for query in query_variants:
            if self.vector_index is None or not self._candidate_index_ready:
                ranked = self.embedding_provider.rank_candidates(query, candidate_texts, purpose=EmbeddingPurpose.LINKING, top_k=min(8, len(candidate_texts)))
            else:
                query_vector = self.embedding_provider.embed_texts([query], purpose=EmbeddingPurpose.QUERY)[0]
                hits = self.vector_index.search(query_vector, namespace=self.vector_namespace, top_k=max(24, min(len(self._vector_alias_strings), len(candidate_texts) * 6)))
                ranked = self._rank_hits_by_canonical_entity(hits, candidate_texts)
                if not ranked:
                    ranked = self.embedding_provider.rank_candidates(query, candidate_texts, purpose=EmbeddingPurpose.LINKING, top_k=min(8, len(candidate_texts)))
            variant_weight = 1.0 if query == query_variants[0] else 0.92
            for match in ranked:
                score = round(match.score * variant_weight, 6)
                existing = aggregated.get(match.candidate_id)
                if existing is None or score > existing.score:
                    aggregated[match.candidate_id] = SimilarityMatch(candidate_id=match.candidate_id, score=score, candidate_text=match.candidate_text, metadata={**match.metadata, 'query_variant': query})
        ranked = sorted(aggregated.values(), key=lambda item: (-item.score, item.candidate_id))[:5]
        self._embedding_rank_cache[cache_key] = ranked
        return ranked

    def _rank_hits_by_canonical_entity(self, hits, candidate_texts):
        grouped: dict[str, SimilarityMatch] = {}
        for hit in hits:
            canonical_entity_id = str(hit.metadata.get('canonical_entity_id') or '')
            if not canonical_entity_id or canonical_entity_id not in candidate_texts:
                continue
            alias_source = str(hit.metadata.get('alias_source') or 'ontology')
            alias_bonus = 0.03 if alias_source != 'ontology' else 0.0
            score = round(min(hit.score + alias_bonus, 0.995), 6)
            existing = grouped.get(canonical_entity_id)
            if existing is None or score > existing.score:
                grouped[canonical_entity_id] = SimilarityMatch(candidate_id=canonical_entity_id, score=score, candidate_text=candidate_texts[canonical_entity_id], metadata={**hit.metadata, 'matched_alias_text': hit.text})
        return sorted(grouped.values(), key=lambda item: (-item.score, item.candidate_id))

    def _unresolved(self, candidate, *, reason, candidate_canonical_ids=None):
        return ResolutionDecision(source_uap_id=candidate.source_uap_id, mention_id=candidate.mention_id, candidate_id=candidate.candidate_id, candidate_text=candidate.text, matched_by='unresolved', confidence=0.0, evidence_span=self._evidence_span(candidate), provider_version=self.embedding_provider.version, candidate_canonical_ids=candidate_canonical_ids or [], unresolved_reason=reason)

    def _prototype_context_support(self, bundle, normalized_context):
        if not normalized_context:
            return 0.0
        phrase_matches = sum((1 for phrase in getattr(bundle, 'related_phrases', ())[:8] if phrase and phrase in normalized_context))
        anchor_matches = sum((1 for phrase in getattr(bundle, 'domain_anchor_phrases', ())[:8] if phrase and phrase in normalized_context))
        neighboring_matches = sum((1 for phrase in (*getattr(bundle, 'neighboring_entity_labels', ())[:6], *getattr(bundle, 'neighboring_aspect_labels', ())[:6], *getattr(bundle, 'neighboring_issue_labels', ())[:6]) if phrase and normalize_alias(phrase) in normalized_context))
        topic_matches = sum((1 for phrase in getattr(bundle, 'related_topic_labels', ())[:4] if phrase and normalize_alias(phrase) in normalized_context))
        score = phrase_matches * 0.03 + anchor_matches * 0.035 + neighboring_matches * 0.02 + topic_matches * 0.025
        return round(min(score, 0.18), 6)

    def _embedding_result_is_ambiguous(self, candidate, ranked_matches):
        if len(ranked_matches) < 2:
            return False
        top = ranked_matches[0]
        second = ranked_matches[1]
        margin = top.score - second.score
        alias_surface_support = _metadata_float(top.metadata.get('alias_surface_support'))
        contextual_support = _metadata_float(top.metadata.get('contextual_support'))
        if alias_surface_support >= 0.58 or contextual_support >= 0.08:
            return False
        if any((method in _HIGH_SIGNAL_DISCOVERY for method in candidate.discovered_by)):
            return False
        required_margin = MIN_EMBEDDING_DISAMBIGUATION_MARGIN
        if candidate.entity_type_hint in {'product', 'product_line'}:
            required_margin += 0.01
        if candidate.entity_type_hint in {'brand', 'organization', 'retailer', 'concept'}:
            required_margin += 0.015
        return top.score >= max(self.embedding_threshold - 0.02, 0.7) and margin < required_margin

    def _decision_for_entity(self, candidate, entity, *, matched_alias, matched_by, confidence, provider_version, candidate_canonical_ids):
        entity_id = str(entity.canonical_entity_id)
        entity_kind = str(entity.entity_kind)
        concept_entity_id = entity_id if entity_kind == 'concept' else None
        canonical_entity_id = entity_id if concept_entity_id is None else None
        return ResolutionDecision(source_uap_id=candidate.source_uap_id, mention_id=candidate.mention_id, candidate_id=candidate.candidate_id, candidate_text=candidate.text, matched_alias=matched_alias, matched_by=matched_by, confidence=confidence, evidence_span=self._evidence_span(candidate), provider_version=provider_version, canonical_entity_id=canonical_entity_id, concept_entity_id=concept_entity_id, resolution_kind='concept' if concept_entity_id is not None else 'canonical_entity', resolved_entity_kind='concept' if concept_entity_id is not None else 'entity', knowledge_layer=entity.knowledge_layer, candidate_canonical_ids=candidate_canonical_ids)

    def _build_alias_token_index(self, aliases):
        index: dict[str, list[EntityAlias]] = {}
        for alias in aliases:
            for token in _significant_tokens(alias.normalized_alias):
                index.setdefault(token, []).append(alias)
        return index

    def _prefilter_aliases(self, normalized_candidate):
        tokens = sorted(_significant_tokens(normalized_candidate), key=len, reverse=True)
        compact_candidate = normalized_candidate.replace(' ', '')
        if not tokens:
            return tuple(self._compact_alias_index.get(compact_candidate, ())) or self._all_aliases
        matched: list[EntityAlias] = []
        seen: set[str] = set()
        for token in tokens[:3]:
            for alias in self._alias_token_index.get(token, []):
                if alias.alias_id in seen:
                    continue
                seen.add(alias.alias_id)
                matched.append(alias)
        if compact_candidate:
            for alias in self._compact_alias_index.get(compact_candidate, ()):
                if alias.alias_id in seen:
                    continue
                seen.add(alias.alias_id)
                matched.append(alias)
        if matched:
            return tuple(matched)
        return tuple((alias for alias in self._all_aliases if abs(len(alias.normalized_alias) - len(normalized_candidate)) <= 12)) or self._all_aliases

    def _build_compact_alias_index(self, aliases):
        index: dict[str, list[EntityAlias]] = {}
        for alias in aliases:
            compact = alias.normalized_alias.replace(' ', '')
            if compact:
                index.setdefault(compact, []).append(alias)
        return index

    def _entity_compact_aliases(self, canonical_entity_id):
        return {alias.normalized_alias.replace(' ', '') for alias in self._all_aliases if alias.canonical_entity_id == canonical_entity_id}

def _ascii_fold(text):
    decomposed = unicodedata.normalize('NFKD', text)
    return ''.join((character for character in decomposed if not unicodedata.combining(character)))

def _query_variants(text):
    normalized = normalize_alias(text)
    ascii_folded = normalize_alias(_ascii_fold(text))
    compact = normalized.replace(' ', '')
    variants = []
    seen = set()
    for candidate in (text.strip(), normalized, ascii_folded, compact):
        cleaned = candidate.strip()
        if not cleaned or cleaned in seen:
            continue
        variants.append(cleaned)
        seen.add(cleaned)
    return variants or [text]

def _metadata_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
