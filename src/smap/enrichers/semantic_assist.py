from __future__ import annotations
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from smap.canonicalization.alias import normalize_alias
from smap.enrichers.semantic_models import SemanticSegment
from smap.providers.base import RerankerProvider, TaxonomyMappingCandidate, TaxonomyMappingProvider
_TOKEN_RE = re.compile('[#@]?[\\w]+', flags=re.UNICODE)
_GENERIC_WINDOW_TOKENS = {'a', 'an', 'and', 'ban', 'cho', 'co', 'cua', 'di', 'ha', 'hay', 'hehe', 'hi', 'hihi', 'khong', 'la', 'lol', 'minh', 'nha', 'nhe', 'nhung', 'oi', 'ok', 'okay', 'qua', 'roi', 'the', 'thi', 'va', 'voi'}

@dataclass(frozen=True, slots=True)
class PhraseWindow:
    text: str
    start: int
    end: int
    normalized_text: str
    variants: tuple[str, ...]
    token_count: int

@dataclass(frozen=True, slots=True)
class SemanticAssistMatch:
    taxonomy_label: str
    source_text: str
    normalized_text: str
    matched_variant: str
    start: int
    end: int
    score: float
    provider_name: str
    provider_model_id: str
    reranked: bool = False
    ambiguity_gap: float = 0.0

def build_phrase_windows(segment, *, max_window_size=4):
    tokens = [(match.group(0), segment.start + match.start(), segment.start + match.end()) for match in _TOKEN_RE.finditer(segment.text)]
    windows = []
    deduped = {}
    for start_index in range(len(tokens)):
        for width in range(1, max_window_size + 1):
            end_index = start_index + width - 1
            if end_index >= len(tokens):
                break
            start = tokens[start_index][1]
            end = tokens[end_index][2]
            raw_text = segment.text[start - segment.start:end - segment.start]
            normalized = normalize_alias(raw_text)
            if len(normalized) < 3:
                continue
            normalized_tokens = tuple((token for token in normalized.split() if token))
            if normalized_tokens and all((token in _GENERIC_WINDOW_TOKENS for token in normalized_tokens)):
                continue
            deduped[start, end] = PhraseWindow(text=raw_text, start=start, end=end, normalized_text=normalized, variants=_variants_for_phrase(raw_text), token_count=width)
    windows.extend(deduped.values())
    return sorted(windows, key=lambda item: (item.start, item.end, item.text))

def collect_mapping_matches(*, segment, taxonomy_labels, taxonomy_mapping_provider, reranker_provider=None, min_score=0.0, phrase_windows=None, mapping_candidate_cache=None):
    if taxonomy_mapping_provider is None:
        return []
    phrase_windows = phrase_windows or build_phrase_windows(segment)
    if not phrase_windows:
        return []
    taxonomy_key = tuple(sorted(taxonomy_labels))
    variant_texts = sorted({variant for window in phrase_windows for variant in window.variants if variant})
    if not variant_texts:
        return []
    mapping_candidate_cache = mapping_candidate_cache if mapping_candidate_cache is not None else {}
    missing_variants = [variant for variant in variant_texts if (taxonomy_key, variant) not in mapping_candidate_cache]
    if missing_variants:
        fresh_candidates = taxonomy_mapping_provider.map_candidates(missing_variants, taxonomy_labels, top_k=4)
        for variant in missing_variants:
            mapping_candidate_cache[taxonomy_key, variant] = list(fresh_candidates.get(variant, []))
    matches = {}
    for window in phrase_windows:
        candidates_with_variant = []
        for variant in window.variants:
            candidates_with_variant.extend(((variant, candidate) for candidate in mapping_candidate_cache.get((taxonomy_key, variant), [])))
        selected = rank_taxonomy_candidates_for_text(text=f'{window.text} || {segment.text}', taxonomy_labels=taxonomy_labels, taxonomy_mapping_provider=taxonomy_mapping_provider, candidates_with_variant=candidates_with_variant, reranker_provider=reranker_provider, top_k=4, mapping_candidate_cache=mapping_candidate_cache)
        if not selected:
            continue
        second_score = selected[1][1].score if len(selected) > 1 else 0.0
        for variant, candidate, reranked in selected:
            score = round(min(candidate.score + min((window.token_count - 1) * 0.03, 0.09), 0.995), 4)
            if score < min_score:
                continue
            match = SemanticAssistMatch(taxonomy_label=candidate.taxonomy_label, source_text=window.text, normalized_text=window.normalized_text, matched_variant=variant, start=window.start, end=window.end, score=score, provider_name=candidate.provider_provenance.provider_name, provider_model_id=candidate.provider_provenance.model_id, reranked=reranked, ambiguity_gap=round(max(score - second_score, 0.0), 4))
            key = (match.taxonomy_label, match.start, match.end)
            existing = matches.get(key)
            if existing is None or match.score > existing.score:
                matches[key] = match
    return sorted(matches.values(), key=lambda item: (-item.score, item.start, item.taxonomy_label))

def mapping_support_by_label(matches):
    grouped = defaultdict(list)
    for match in matches:
        grouped[match.taxonomy_label].append(match)
    return {label: sorted(items, key=lambda item: (-item.score, item.start, item.source_text)) for label, items in grouped.items()}

def rank_taxonomy_candidates_for_text(*, text, taxonomy_labels, taxonomy_mapping_provider, reranker_provider, candidates_with_variant=None, top_k=4, mapping_candidate_cache=None):
    if taxonomy_mapping_provider is None:
        return []
    candidate_pairs = list(candidates_with_variant or [])
    if not candidate_pairs:
        taxonomy_key = tuple(sorted(taxonomy_labels))
        mapping_candidate_cache = mapping_candidate_cache if mapping_candidate_cache is not None else {}
        variants = [variant for variant in _variants_for_phrase(text) if variant]
        missing_variants = [variant for variant in variants if (taxonomy_key, variant) not in mapping_candidate_cache]
        if missing_variants:
            mapped = taxonomy_mapping_provider.map_candidates(missing_variants, taxonomy_labels, top_k=min(max(top_k, 3), max(len(taxonomy_labels), 1)))
            for variant in missing_variants:
                mapping_candidate_cache[taxonomy_key, variant] = list(mapped.get(variant, []))
        for variant in variants:
            candidate_pairs.extend(((variant, candidate) for candidate in mapping_candidate_cache.get((taxonomy_key, variant), [])))
    if not candidate_pairs:
        return []
    best_by_label = {}
    for variant, candidate in candidate_pairs:
        current = best_by_label.get(candidate.taxonomy_label)
        if current is None or candidate.score > current[1].score:
            best_by_label[candidate.taxonomy_label] = (variant, candidate)
    if not best_by_label:
        return []
    rerank_bonus = {}
    if reranker_provider is not None and len(best_by_label) > 1:
        reranked = reranker_provider.rerank(text, [label.replace('_', ' ') for label in best_by_label])
        order = {match.candidate_id.replace(' ', '_'): index for index, match in enumerate(reranked)}
        for label in best_by_label:
            rank_index = order.get(label, len(best_by_label))
            rerank_bonus[label] = max(0.0, (len(best_by_label) - rank_index) * 0.015)
    ranked = []
    for label, (variant, candidate) in best_by_label.items():
        adjusted = replace(candidate, score=round(min(candidate.score + rerank_bonus.get(label, 0.0), 0.995), 6))
        ranked.append((variant, adjusted, rerank_bonus.get(label, 0.0) > 0.0))
    return sorted(ranked, key=lambda item: (-item[1].score, item[1].taxonomy_label))[:top_k]

def _variants_for_phrase(text):
    normalized = normalize_alias(text)
    stripped = normalize_alias(text.lstrip('#@'))
    ascii_folded = _ascii_fold(text)
    ascii_normalized = normalize_alias(ascii_folded)
    compact = normalized.replace(' ', '')
    repeated_collapsed = normalize_alias(_collapse_repeated_characters(text))
    compact_repeated = repeated_collapsed.replace(' ', '')
    slash_split = normalize_alias(text.replace('/', ' ').replace('-', ' '))
    candidates = [text.strip(), normalized, stripped, ascii_folded, ascii_normalized, compact if len(compact) >= 4 else '', repeated_collapsed, compact_repeated if len(compact_repeated) >= 4 else '', slash_split]
    deduped = []
    seen = set()
    for candidate in candidates:
        cleaned = candidate.strip()
        if not cleaned or cleaned in seen:
            continue
        deduped.append(cleaned)
        seen.add(cleaned)
    return tuple(deduped)

def _ascii_fold(text):
    decomposed = unicodedata.normalize('NFKD', text)
    return ''.join((character for character in decomposed if not unicodedata.combining(character)))

def _collapse_repeated_characters(text):
    return re.sub('(.)\\1{2,}', '\\1\\1', text, flags=re.IGNORECASE)
