from __future__ import annotations
import unicodedata
from dataclasses import dataclass, field
from smap.enrichers.anchors import ASPECT_SEEDS, ISSUE_SEEDS
from smap.providers.base import EmbeddingProvider, EmbeddingPurpose, ProviderProvenance, RerankerProvider, SimilarityMatch, TaxonomyMappingCandidate, TaxonomyMappingProvider
_GENERIC_TAXONOMY_PROTOTYPES = {'usability': ('user flow', 'setup flow', 'navigation', 'interface', 'checkout flow', 'de su dung', 'luong thao tac'), 'usability_problem': ('user flow issue', 'setup issue', 'navigation problem', 'interface problem', 'checkout friction', 'kho dung', 'roi flow'), 'customer_service': ('support quality', 'response quality', 'staff support', 'response team', 'help desk', 'after sales team', 'after sales desk', 'ho tro', 'cskh'), 'service_problem': ('slow support', 'late response', 'slow response team', 'help desk issue', 'service issue', 'ho tro cham', 'phan hoi cham'), 'performance': ('speed', 'loading', 'stability', 'lag', 'do tre'), 'performance_problem': ('slow loading', 'performance issue', 'stability issue', 'bi lag', 'qua cham', 'tre'), 'delivery': ('shipping speed', 'delivery timing', 'giao hang'), 'delivery_problem': ('shipping delay', 'delivery delay', 'ship cham', 'giao tre'), 'pricing_value_concern': ('too expensive', 'poor value', 'price concern'), 'trust_concern': ('authenticity', 'credibility concern', 'trust issue', 'khong tin', 'fake')}

@dataclass(slots=True)
class EmbeddingTaxonomyMappingProvider(TaxonomyMappingProvider):
    embedding_provider: EmbeddingProvider
    version: str = 'embedding-taxonomy-map-v2'
    provenance: ProviderProvenance = field(init=False)
    _prototype_cache: dict[tuple[str, ...], dict[str, tuple[str, str]]] = field(default_factory=dict, init=False)
    _prototype_vector_cache: dict[tuple[str, ...], dict[str, tuple[float, ...]]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.provenance = ProviderProvenance(provider_kind='semantic_assist', provider_name='embedding_taxonomy_mapping', provider_version=self.version, model_id=self.embedding_provider.provenance.model_id, device=self.embedding_provider.provenance.device)

    def map_labels(self, labels, taxonomy_labels):
        candidates = self.map_candidates(labels, taxonomy_labels, top_k=1)
        return {label: mapped[0].taxonomy_label for label, mapped in candidates.items() if mapped}

    def map_candidates(self, labels, taxonomy_labels, *, top_k=3):
        taxonomy_key = tuple(sorted(taxonomy_labels))
        prototype_texts = self._prototype_cache.get(taxonomy_key)
        prototype_vectors = self._prototype_vector_cache.get(taxonomy_key)
        if prototype_texts is None or prototype_vectors is None:
            prototype_texts = {}
            for taxonomy_label in taxonomy_key:
                for index, prototype in enumerate(_taxonomy_prototypes(taxonomy_label)):
                    prototype_texts[f'{taxonomy_label}::{index}'] = (taxonomy_label, prototype)
            prototype_ids = list(prototype_texts)
            prototype_embeddings = self.embedding_provider.embed_texts([prototype_texts[prototype_id][1] for prototype_id in prototype_ids], purpose=EmbeddingPurpose.PASSAGE)
            prototype_vectors = dict(zip(prototype_ids, prototype_embeddings, strict=True))
            self._prototype_cache[taxonomy_key] = prototype_texts
            self._prototype_vector_cache[taxonomy_key] = prototype_vectors
        unique_labels = list(dict.fromkeys(labels))
        label_vectors = dict(zip(unique_labels, self.embedding_provider.embed_texts(unique_labels, purpose=EmbeddingPurpose.LINKING), strict=True))
        mapped: dict[str, list[TaxonomyMappingCandidate]] = {}
        for label in unique_labels:
            query_vector = label_vectors.get(label)
            ranked = []
            if query_vector is not None:
                ranked = sorted([SimilarityMatch(candidate_id=prototype_id, score=round(sum((a * b for a, b in zip(query_vector, prototype_vectors[prototype_id], strict=True))), 6), candidate_text=prototype_texts[prototype_id][1]) for prototype_id in prototype_texts], key=lambda item: (-item.score, item.candidate_id))
            by_taxonomy: dict[str, TaxonomyMappingCandidate] = {}
            for match in ranked:
                prototype_entry = prototype_texts.get(match.candidate_id)
                if prototype_entry is None:
                    continue
                taxonomy_label, prototype_text = prototype_entry
                lexical_score = _lexical_similarity(label, prototype_text)
                compact_bonus = 0.04 if _compact(label) == _compact(prototype_text) else 0.0
                blended_score = round(max(match.score, 0.0) * 0.45 + lexical_score * 0.55 + compact_bonus, 6)
                existing = by_taxonomy.get(taxonomy_label)
                candidate = TaxonomyMappingCandidate(source_text=label, taxonomy_label=taxonomy_label, score=blended_score, provider_provenance=self.provenance, metadata={'candidate_text': match.candidate_text, 'prototype_text': prototype_text, 'embedding_score': match.score, 'lexical_score': lexical_score})
                if existing is None or candidate.score > existing.score:
                    by_taxonomy[taxonomy_label] = candidate
            mapped[label] = sorted(by_taxonomy.values(), key=lambda item: (-item.score, item.taxonomy_label))[:top_k]
        return {label: mapped.get(label, []) for label in labels}

@dataclass(slots=True)
class EmbeddingRerankerProvider(RerankerProvider):
    embedding_provider: EmbeddingProvider
    version: str = 'embedding-reranker-v1'
    provenance: ProviderProvenance = field(init=False)

    def __post_init__(self):
        self.provenance = ProviderProvenance(provider_kind='semantic_assist', provider_name='embedding_reranker', provider_version=self.version, model_id=self.embedding_provider.provenance.model_id, device=self.embedding_provider.provenance.device)

    def rerank(self, query, candidates):
        candidate_map = {str(index): candidate for index, candidate in enumerate(candidates)}
        ranked = self.embedding_provider.rank_candidates(query, candidate_map, purpose=EmbeddingPurpose.LINKING, top_k=len(candidates))
        return [SimilarityMatch(candidate_id=candidate_map.get(match.candidate_id, match.candidate_id), score=match.score, candidate_text=candidate_map.get(match.candidate_id, match.candidate_id), metadata=match.metadata) for match in ranked]

def _taxonomy_prototypes(taxonomy_label):
    phrases = [taxonomy_label.replace('_', ' ')]
    if taxonomy_label in ASPECT_SEEDS:
        phrases.extend(ASPECT_SEEDS[taxonomy_label])
    if taxonomy_label in ISSUE_SEEDS:
        phrases.extend(ISSUE_SEEDS[taxonomy_label])
    phrases.extend(_GENERIC_TAXONOMY_PROTOTYPES.get(taxonomy_label, ()))
    deduped = []
    seen = set()
    for phrase in phrases:
        cleaned = phrase.strip()
        if not cleaned:
            continue
        for variant in _prototype_variants(cleaned):
            folded = variant.casefold()
            if folded in seen:
                continue
            deduped.append(variant)
            seen.add(folded)
    return tuple(deduped)

def _lexical_similarity(left, right):
    left_tokens = set(_normalized_variant(left).split())
    right_tokens = set(_normalized_variant(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    if _compact(left) == _compact(right):
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)

def _prototype_variants(text):
    normalized = _normalized_variant(text)
    ascii_folded = _ascii_fold(normalized)
    compact = _compact(normalized)
    variants = []
    seen = set()
    for candidate in (text.strip(), normalized, ascii_folded, compact):
        cleaned = candidate.strip()
        if not cleaned or cleaned in seen:
            continue
        variants.append(cleaned)
        seen.add(cleaned)
    return tuple(variants)

def _normalized_variant(text):
    return ' '.join(text.casefold().replace('_', ' ').replace('-', ' ').split())

def _compact(text):
    return _normalized_variant(text).replace(' ', '')

def _ascii_fold(text):
    decomposed = unicodedata.normalize('NFKD', text)
    return ''.join((character for character in decomposed if not unicodedata.combining(character)))
