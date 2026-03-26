from __future__ import annotations
import unicodedata
from dataclasses import dataclass, field
from smap.canonicalization.alias import normalize_alias
from smap.canonicalization.models import DiscoveryMethod, EntityCandidate
from smap.providers.base import EmbeddingProvider, EmbeddingPurpose, ProviderMetadata, RecognizedEntitySpan

def _compatible_entity_type(left, right):
    return left is None or right is None or left == right

def _spans_overlap(left, right):
    if left.start_char is None or left.end_char is None or right.start_char is None or (right.end_char is None):
        return False
    return left.start_char < right.end_char and right.start_char < left.end_char

def _span_length(candidate):
    if candidate.start_char is None or candidate.end_char is None:
        return 0
    return max(candidate.end_char - candidate.start_char, 0)

def _compact_text(candidate):
    normalized = candidate.normalized_text or normalize_alias(candidate.text)
    return normalized.replace(' ', '')

def _ascii_compact_text(candidate):
    return _ascii_fold(candidate.normalized_text or normalize_alias(candidate.text)).replace(' ', '')

def _token_overlap(left, right):
    left_tokens = set((left.normalized_text or normalize_alias(left.text)).split())
    right_tokens = set((right.normalized_text or normalize_alias(right.text)).split())
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)

@dataclass(slots=True)
class MergedCandidateTrace:
    normalized_text: str
    entity_type_hint: str | None
    sources: list[str] = field(default_factory=list)

class CandidateMerger:

    def __init__(self, *, embedding_provider=None, semantic_merge_threshold=0.82):
        self.embedding_provider = embedding_provider
        self.semantic_merge_threshold = semantic_merge_threshold
        self._semantic_similarity_cache: dict[tuple[str, str], float] = {}

    def merge(self, *, base_candidates, recognized_spans, mention_id, source_uap_id, context_text, surrounding_text):
        merged = list(base_candidates)
        for span in recognized_spans:
            span_candidate = EntityCandidate(candidate_id=f'{mention_id}:{span.provider_provenance.provider_name}:{span.start}:{span.end}', source_uap_id=source_uap_id, mention_id=mention_id, text=span.text, normalized_text=span.normalized_text or normalize_alias(span.text), start_char=span.start, end_char=span.end, entity_type_hint=span.entity_type_hint, confidence=round(span.confidence, 3), discovered_by=[self._discovery_method(span.provider_provenance.provider_name, span.metadata)], evidence_mention_ids=[mention_id], context_text=context_text, surrounding_text=surrounding_text, full_text=surrounding_text)
            self._merge_one(merged, span_candidate)
        return sorted(merged, key=lambda item: (-item.confidence, item.start_char if item.start_char is not None else 10 ** 9, item.text))

    def _merge_one(self, merged, candidate):
        for index, existing in enumerate(merged):
            if self._groupable(existing, candidate):
                prefer_candidate_surface = _span_length(candidate) > _span_length(existing)
                merged_sources = sorted(set(existing.discovered_by + candidate.discovered_by))
                source_bonus = min(max(len(merged_sources) - 1, 0) * 0.01, 0.03)
                merged[index] = existing.model_copy(update={'confidence': round(min(max(existing.confidence, candidate.confidence) + 0.02 + source_bonus, 0.98), 3), 'text': candidate.text if prefer_candidate_surface else existing.text, 'normalized_text': candidate.normalized_text if prefer_candidate_surface else existing.normalized_text, 'entity_type_hint': candidate.entity_type_hint if prefer_candidate_surface and candidate.entity_type_hint is not None else existing.entity_type_hint or candidate.entity_type_hint, 'start_char': candidate.start_char if prefer_candidate_surface and candidate.start_char is not None else existing.start_char if existing.start_char is not None else candidate.start_char, 'end_char': candidate.end_char if prefer_candidate_surface and candidate.end_char is not None else existing.end_char if existing.end_char is not None else candidate.end_char, 'surrounding_text': candidate.surrounding_text or existing.surrounding_text, 'context_text': existing.context_text or candidate.context_text, 'full_text': existing.full_text or candidate.full_text, 'discovered_by': merged_sources})
                return
        merged.append(candidate)

    def _groupable(self, left, right):
        same_text = (left.normalized_text or normalize_alias(left.text)) == (right.normalized_text or normalize_alias(right.text))
        compact_match = _compact_text(left) == _compact_text(right)
        semantic_duplicate = _token_overlap(left, right) >= 0.85 or ((left.normalized_text or normalize_alias(left.text)) in (right.normalized_text or normalize_alias(right.text)) or (right.normalized_text or normalize_alias(right.text)) in (left.normalized_text or normalize_alias(left.text)))
        ascii_compact_match = _ascii_compact_text(left) == _ascii_compact_text(right)
        semantic_similarity = self._semantic_duplicate(left, right)
        return _compatible_entity_type(left.entity_type_hint, right.entity_type_hint) and (same_text or compact_match or ascii_compact_match or _spans_overlap(left, right) or semantic_duplicate or semantic_similarity)

    def _semantic_duplicate(self, left, right):
        if self.embedding_provider is None:
            return False
        left_text = left.normalized_text or normalize_alias(left.text)
        right_text = right.normalized_text or normalize_alias(right.text)
        if not left_text or not right_text:
            return False
        cache_key = (left_text, right_text) if left_text <= right_text else (right_text, left_text)
        cached = self._semantic_similarity_cache.get(cache_key)
        if cached is not None:
            return cached >= self.semantic_merge_threshold
        lexical_overlap = _token_overlap(left, right)
        shared_digit = any((character.isdigit() for character in left_text)) and any((character.isdigit() for character in right_text))
        if lexical_overlap < 0.34 and (not shared_digit):
            return False
        vectors = self.embedding_provider.embed_texts([left_text, right_text], purpose=EmbeddingPurpose.LINKING)
        if len(vectors) != 2:
            return False
        similarity = sum((a * b for a, b in zip(vectors[0], vectors[1], strict=True)))
        self._semantic_similarity_cache[cache_key] = similarity
        return similarity >= self.semantic_merge_threshold

    def _discovery_method(self, provider_name, metadata):
        if provider_name == 'phobert_ner':
            return 'phobert_ner'
        source = str(metadata.get('source') or '')
        if 'handle' in source:
            return 'handle_pattern'
        return 'rule_ruler'

def _ascii_fold(text):
    decomposed = unicodedata.normalize('NFKD', text)
    return ''.join((character for character in decomposed if not unicodedata.combining(character)))
