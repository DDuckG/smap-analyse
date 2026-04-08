from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from smap.canonicalization.alias import AliasRegistry, boundary_spans, normalize_alias
from smap.canonicalization.models import DiscoveryMethod, EntityAlias, EntityCandidate
from smap.normalization.models import MentionRecord
from smap.threads.models import MentionContext

PRODUCT_CODE_RE = re.compile(r"\b[a-z]{1,10}[- ]?\d{1,4}[a-z0-9-]{0,4}\b", re.IGNORECASE)
TITLE_SPAN_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}|[A-Z][a-z]+[A-Z][A-Za-z0-9]+)\b"
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
RAW_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{1,}")

STOPWORDS = {
    "a",
    "an",
    "and",
    "anh",
    "ban",
    "banh",
    "bạn",
    "but",
    "cho",
    "co",
    "cua",
    "của",
    "for",
    "hay",
    "khong",
    "không",
    "la",
    "là",
    "mai",
    "nay",
    "này",
    "new",
    "qua",
    "quá",
    "review",
    "roi",
    "rồi",
    "the",
    "this",
    "voi",
    "với",
}
REFERENTIAL_CUES = {
    "bản này",
    "con này",
    "dòng này",
    "em này",
    "mau nay",
    "mẫu này",
    "that one",
    "this one",
}
_TYPE_CUES: dict[str, tuple[str, ...]] = {
    "brand": ("brand", "hãng", "thuong hieu", "thương hiệu", "nhãn hàng"),
    "product": ("product", "model", "variant", "gel", "cleanser", "serum", "foam"),
    "organization": ("organization", "company", "tap doan", "tập đoàn", "cong ty", "công ty"),
    "retailer": ("retailer", "official mall", "dealer", "showroom", "store", "cua hang", "cửa hàng"),
    "facility": ("facility", "factory", "plant", "nha may", "nhà máy", "xuong", "xưởng"),
    "location": ("location", "city", "province", "thi truong", "thị trường", "quoc gia", "quốc gia"),
    "concept": ("concept", "routine", "category", "phan khuc", "phân khúc"),
}
_OPEN_TYPE_PRIORITY = ("product", "brand", "retailer", "organization", "facility", "location", "concept")


def _significant_tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(text)
        if token not in STOPWORDS and (len(token) >= 3 or any(char.isdigit() for char in token))
    ]


def _decompose_hashtag(hashtag: str) -> str:
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", hashtag).replace("_", " ").replace("-", " ")
    normalized = normalize_alias(parts)
    return normalized


def _candidate_slug(text: str) -> str:
    slug = normalize_alias(text).replace(" ", "_")
    return slug[:80] if slug else "candidate"


class EntityCandidateDiscoverer:
    def __init__(
        self,
        alias_registry: AliasRegistry,
        *,
        repeated_phrase_min_mentions: int = 2,
    ) -> None:
        self.alias_registry = alias_registry
        self.repeated_phrase_min_mentions = repeated_phrase_min_mentions
        self._repeated_phrases: dict[str, int] = {}
        self._all_aliases = tuple(self.alias_registry.all_aliases())
        self._alias_token_index = self._build_alias_token_index()
        self._compact_alias_index = self._build_compact_alias_index()

    def prepare(
        self,
        mentions: list[MentionRecord],
        contexts: list[MentionContext],
    ) -> None:
        del contexts
        phrase_mentions: dict[str, set[str]] = defaultdict(set)
        for mention in mentions:
            for phrase in self._iter_phrase_candidates(mention.normalized_text_compact):
                if phrase in self.alias_registry.aliases_by_normalized:
                    continue
                if phrase in self.alias_registry.noise_terms:
                    continue
                phrase_mentions[phrase].add(mention.mention_id)
        self._repeated_phrases = {
            phrase: len(mention_ids)
            for phrase, mention_ids in phrase_mentions.items()
            if len(mention_ids) >= self.repeated_phrase_min_mentions
        }

    def discover(
        self,
        mention: MentionRecord,
        context: MentionContext | None,
    ) -> list[EntityCandidate]:
        candidates: dict[tuple[str, str | None], EntityCandidate] = {}

        for candidate in self._discover_alias_surface_matches(mention, context):
            self._merge_candidate(candidates, candidate)
        for candidate in self._discover_brand_context_candidates(mention, context):
            self._merge_candidate(candidates, candidate)
        for candidate in self._discover_hashtags(mention, context):
            self._merge_candidate(candidates, candidate)
        for candidate in self._discover_product_codes(mention, context):
            self._merge_candidate(candidates, candidate)
        for candidate in self._discover_title_spans(mention, context):
            self._merge_candidate(candidates, candidate)
        for candidate in self._discover_repeated_phrases(mention, context):
            self._merge_candidate(candidates, candidate)
        for candidate in self._discover_token_ngrams(mention, context):
            self._merge_candidate(candidates, candidate)
        for candidate in self._discover_context_spans(mention, context, have_local_candidates=bool(candidates)):
            self._merge_candidate(candidates, candidate)

        return sorted(
            candidates.values(),
            key=lambda item: (-item.confidence, item.start_char if item.start_char is not None else 10**9, item.text),
        )

    def _iter_phrase_candidates(self, normalized_text: str) -> Iterable[str]:
        tokens = _significant_tokens(normalized_text)
        token_count = len(tokens)
        for size in (1, 2, 3):
            if token_count < size:
                continue
            for index in range(token_count - size + 1):
                phrase_tokens = tokens[index : index + size]
                if size == 1 and len(phrase_tokens[0]) < 5 and not any(
                    character.isdigit() for character in phrase_tokens[0]
                ):
                    continue
                phrase = " ".join(phrase_tokens)
                if phrase in self.alias_registry.noise_terms:
                    continue
                yield phrase

    def _discover_alias_surface_matches(
        self,
        mention: MentionRecord,
        context: MentionContext | None,
    ) -> list[EntityCandidate]:
        results: list[EntityCandidate] = []
        for alias in self._surface_aliases_for_text(mention.normalized_text_compact):
            for start_char, end_char in boundary_spans(mention.raw_text, alias.alias):
                entity = self.alias_registry.entities[alias.canonical_entity_id]
                surface = mention.raw_text[start_char:end_char]
                results.append(
                    self._build_candidate(
                        mention=mention,
                        context=context,
                        text=surface,
                        normalized_text=normalize_alias(surface),
                        start_char=start_char,
                        end_char=end_char,
                        entity_type_hint=entity.entity_type,
                        confidence=0.95,
                        discovered_by="alias_scan",
                    )
                )
        return results

    def _discover_brand_context_candidates(
        self,
        mention: MentionRecord,
        context: MentionContext | None,
    ) -> list[EntityCandidate]:
        results: list[EntityCandidate] = []
        for alias in self._surface_aliases_for_text(mention.normalized_text_compact):
            entity = self.alias_registry.entities[alias.canonical_entity_id]
            if entity.entity_type != "brand":
                continue
            for _, end_char in boundary_spans(mention.raw_text, alias.alias):
                trailing_tokens = RAW_TOKEN_RE.findall(mention.raw_text[end_char:])
                product_tokens = [
                    token
                    for token in trailing_tokens[:3]
                    if token.lower() not in STOPWORDS
                    and (token[0].isupper() or any(character.isdigit() for character in token))
                ]
                if not product_tokens:
                    continue
                candidate_text = " ".join([alias.alias, *product_tokens[:2]])
                if normalize_alias(candidate_text) in self.alias_registry.aliases_by_normalized:
                    continue
                start_char = mention.raw_text.casefold().find(alias.alias.casefold())
                results.append(
                    self._build_candidate(
                        mention=mention,
                        context=context,
                        text=candidate_text,
                        normalized_text=normalize_alias(candidate_text),
                        start_char=start_char if start_char >= 0 else None,
                        end_char=(start_char + len(candidate_text)) if start_char >= 0 else None,
                        entity_type_hint="product",
                        confidence=0.62,
                        discovered_by="brand_context",
                    )
                )
        return results

    def _discover_hashtags(
        self,
        mention: MentionRecord,
        context: MentionContext | None,
    ) -> list[EntityCandidate]:
        results: list[EntityCandidate] = []
        for hashtag in mention.hashtags:
            normalized_tag = _decompose_hashtag(hashtag)
            if len(normalized_tag) < 3 or self.alias_registry.is_noise_term(normalized_tag):
                continue
            if not self._supports_open_hashtag_candidate(hashtag):
                continue
            start_char = mention.raw_text.casefold().find(f"#{hashtag.casefold()}")
            end_char = start_char + len(hashtag) + 1 if start_char >= 0 else None
            results.append(
                self._build_candidate(
                    mention=mention,
                    context=context,
                    text=normalized_tag if " " in normalized_tag else hashtag,
                    normalized_text=normalized_tag,
                    start_char=start_char if start_char >= 0 else None,
                    end_char=end_char,
                    entity_type_hint=None,
                    confidence=0.52,
                    discovered_by="hashtag",
                )
            )
        return results

    def _discover_product_codes(
        self,
        mention: MentionRecord,
        context: MentionContext | None,
    ) -> list[EntityCandidate]:
        results: list[EntityCandidate] = []
        for match in PRODUCT_CODE_RE.finditer(mention.raw_text):
            code = match.group(0)
            results.append(
                self._build_candidate(
                    mention=mention,
                    context=context,
                    text=code,
                    normalized_text=normalize_alias(code),
                    start_char=match.start(),
                    end_char=match.end(),
                    entity_type_hint="product",
                    confidence=0.68,
                    discovered_by="product_code",
                )
            )
        return results

    def _discover_title_spans(
        self,
        mention: MentionRecord,
        context: MentionContext | None,
    ) -> list[EntityCandidate]:
        results: list[EntityCandidate] = []
        for match in TITLE_SPAN_RE.finditer(mention.raw_text):
            surface = match.group(0).strip()
            normalized_surface = normalize_alias(surface)
            if len(normalized_surface) < 4 or self.alias_registry.is_noise_term(normalized_surface):
                continue
            entity_type_hint = self._infer_open_entity_type(
                normalized_surface,
                surface=surface,
                full_text=mention.normalized_text_compact,
            )
            if entity_type_hint is None and not self._supports_open_title_candidate(surface, normalized_surface):
                continue
            results.append(
                self._build_candidate(
                    mention=mention,
                    context=context,
                    text=surface,
                    normalized_text=normalized_surface,
                    start_char=match.start(),
                    end_char=match.end(),
                    entity_type_hint=entity_type_hint,
                    confidence=0.64 if entity_type_hint is not None else 0.54,
                    discovered_by="title_span",
                )
            )
        return results

    def _discover_repeated_phrases(
        self,
        mention: MentionRecord,
        context: MentionContext | None,
    ) -> list[EntityCandidate]:
        results: list[EntityCandidate] = []
        lowered_compact = mention.normalized_text_compact.casefold()
        for phrase, mention_count in self._repeated_phrases.items():
            if phrase not in lowered_compact:
                continue
            entity_type_hint = self._infer_open_entity_type(
                phrase,
                surface=phrase,
                full_text=mention.normalized_text_compact,
            )
            if entity_type_hint is None and not self._supports_open_phrase_candidate(phrase):
                continue
            start_char = mention.raw_text.casefold().find(phrase)
            end_char = (start_char + len(phrase)) if start_char >= 0 else None
            results.append(
                self._build_candidate(
                    mention=mention,
                    context=context,
                    text=phrase,
                    normalized_text=phrase,
                    start_char=start_char if start_char >= 0 else None,
                    end_char=end_char,
                    entity_type_hint=entity_type_hint,
                    confidence=min((0.48 if entity_type_hint is not None else 0.42) + (mention_count * 0.05), 0.7),
                    discovered_by="repeated_phrase",
                )
            )
        return results

    def _discover_token_ngrams(
        self,
        mention: MentionRecord,
        context: MentionContext | None,
    ) -> list[EntityCandidate]:
        results: list[EntityCandidate] = []
        for phrase in self._iter_phrase_candidates(mention.normalized_text_compact):
            if phrase in self._repeated_phrases:
                continue
            if phrase in self.alias_registry.aliases_by_normalized:
                continue
            if phrase in self.alias_registry.noise_terms:
                continue
            if " " not in phrase:
                continue
            entity_type_hint = self._infer_open_entity_type(
                phrase,
                surface=phrase,
                full_text=mention.normalized_text_compact,
            )
            if entity_type_hint is None:
                continue
            if not self._supports_open_ngram_candidate(mention, phrase):
                continue
            start_char = mention.raw_text.casefold().find(phrase)
            end_char = (start_char + len(phrase)) if start_char >= 0 else None
            results.append(
                self._build_candidate(
                    mention=mention,
                    context=context,
                    text=phrase,
                    normalized_text=phrase,
                    start_char=start_char if start_char >= 0 else None,
                    end_char=end_char,
                    entity_type_hint=entity_type_hint,
                    confidence=0.4,
                    discovered_by="token_ngram",
                )
            )
        return results

    def _supports_open_ngram_candidate(self, mention: MentionRecord, phrase: str) -> bool:
        if any(character.isdigit() for character in phrase):
            return True
        raw_lower = mention.raw_text.casefold()
        start_char = raw_lower.find(phrase)
        if start_char < 0:
            return False
        raw_surface = mention.raw_text[start_char : start_char + len(phrase)]
        supportive_tokens = [
            token
            for token in RAW_TOKEN_RE.findall(raw_surface)
            if any(character.isupper() for character in token) or any(character.isdigit() for character in token)
        ]
        return len(supportive_tokens) >= 2

    def _supports_open_hashtag_candidate(self, hashtag: str) -> bool:
        return any(character.isupper() for character in hashtag) or any(character.isdigit() for character in hashtag)

    def _discover_context_spans(
        self,
        mention: MentionRecord,
        context: MentionContext | None,
        *,
        have_local_candidates: bool,
    ) -> list[EntityCandidate]:
        if context is None:
            return []
        if have_local_candidates and not self._looks_referential(mention.normalized_text_compact):
            return []

        context_texts = [item for item in (context.parent_text, context.root_text) if item]
        results: list[EntityCandidate] = []
        for context_text in context_texts:
            for match in TITLE_SPAN_RE.finditer(context_text):
                surface = match.group(0).strip()
                normalized_surface = normalize_alias(surface)
                if len(normalized_surface) < 4 or self.alias_registry.is_noise_term(normalized_surface):
                    continue
                entity_type_hint = self._infer_open_entity_type(
                    normalized_surface,
                    surface=surface,
                    full_text=normalize_alias(context_text),
                )
                if entity_type_hint is None:
                    continue
                results.append(
                    self._build_candidate(
                        mention=mention,
                        context=context,
                        text=surface,
                        normalized_text=normalized_surface,
                        start_char=None,
                        end_char=None,
                        entity_type_hint=entity_type_hint,
                        confidence=0.46,
                        discovered_by="context_span",
                    )
                )
            for match in PRODUCT_CODE_RE.finditer(context_text):
                surface = match.group(0)
                results.append(
                    self._build_candidate(
                        mention=mention,
                        context=context,
                        text=surface,
                        normalized_text=normalize_alias(surface),
                        start_char=None,
                        end_char=None,
                        entity_type_hint="product",
                        confidence=0.49,
                        discovered_by="context_span",
                    )
                )
        return results

    def _looks_referential(self, normalized_text: str) -> bool:
        return any(cue in normalized_text for cue in REFERENTIAL_CUES)

    def _build_candidate(
        self,
        *,
        mention: MentionRecord,
        context: MentionContext | None,
        text: str,
        normalized_text: str,
        start_char: int | None,
        end_char: int | None,
        entity_type_hint: str | None,
        confidence: float,
        discovered_by: DiscoveryMethod,
    ) -> EntityCandidate:
        surrounding_text = self._surrounding_text(mention.raw_text, start_char, end_char)
        return EntityCandidate(
            candidate_id=f"{mention.mention_id}:{_candidate_slug(normalized_text or text)}",
            source_uap_id=mention.source_uap_id,
            mention_id=mention.mention_id,
            text=text,
            normalized_text=normalized_text,
            start_char=start_char,
            end_char=end_char,
            entity_type_hint=entity_type_hint,
            confidence=round(confidence, 3),
            discovered_by=[discovered_by],
            evidence_mention_ids=[mention.mention_id],
            context_text=context.context_text if context else None,
            surrounding_text=surrounding_text if surrounding_text else (context.context_text if context else None),
            full_text=mention.raw_text,
        )

    def _merge_candidate(
        self,
        candidates: dict[tuple[str, str | None], EntityCandidate],
        candidate: EntityCandidate,
    ) -> None:
        key = (candidate.normalized_text or normalize_alias(candidate.text), candidate.entity_type_hint)
        existing = candidates.get(key)
        if existing is None:
            candidates[key] = candidate
            return

        merged_methods = sorted(set(existing.discovered_by + candidate.discovered_by))
        merged_mentions = sorted(set(existing.evidence_mention_ids + candidate.evidence_mention_ids))
        candidates[key] = existing.model_copy(
            update={
                "confidence": round(min(max(existing.confidence, candidate.confidence) + 0.03, 0.96), 3),
                "discovered_by": merged_methods,
                "evidence_mention_ids": merged_mentions,
                "start_char": existing.start_char if existing.start_char is not None else candidate.start_char,
                "end_char": existing.end_char if existing.end_char is not None else candidate.end_char,
                "surrounding_text": existing.surrounding_text or candidate.surrounding_text,
                "context_text": existing.context_text or candidate.context_text,
            }
        )

    def _surrounding_text(
        self,
        raw_text: str,
        start_char: int | None,
        end_char: int | None,
    ) -> str | None:
        if start_char is None or end_char is None:
            return None
        window_start = max(start_char - 32, 0)
        window_end = min(end_char + 32, len(raw_text))
        return raw_text[window_start:window_end]

    def _build_alias_token_index(self) -> dict[str, list[EntityAlias]]:
        index: dict[str, list[EntityAlias]] = {}
        for alias in self._all_aliases:
            for token in _significant_tokens(alias.normalized_alias):
                index.setdefault(token, []).append(alias)
        return index

    def _surface_aliases_for_text(self, normalized_text: str) -> tuple[EntityAlias, ...]:
        tokens = sorted(set(_significant_tokens(normalized_text)), key=len, reverse=True)
        compact_text = normalized_text.replace(" ", "")
        if not tokens:
            return tuple(self._compact_alias_index.get(compact_text, ())) or self._all_aliases
        matched: list[EntityAlias] = []
        seen: set[str] = set()
        for token in tokens[:4]:
            for alias in self._alias_token_index.get(token, []):
                if alias.alias_id in seen:
                    continue
                seen.add(alias.alias_id)
                matched.append(alias)
        if compact_text:
            for alias in self._compact_alias_index.get(compact_text, ()):
                if alias.alias_id in seen:
                    continue
                seen.add(alias.alias_id)
                matched.append(alias)
        return tuple(matched) if matched else self._all_aliases

    def _build_compact_alias_index(self) -> dict[str, list[EntityAlias]]:
        index: dict[str, list[EntityAlias]] = {}
        for alias in self._all_aliases:
            compact = alias.normalized_alias.replace(" ", "")
            if compact:
                index.setdefault(compact, []).append(alias)
        return index

    def _supports_open_title_candidate(self, surface: str, normalized_surface: str) -> bool:
        tokens = [token for token in normalized_surface.split() if token]
        if any(character.isdigit() for character in normalized_surface):
            return True
        return len(tokens) >= 2 and any(character.isupper() for character in surface)

    def _supports_open_phrase_candidate(self, phrase: str) -> bool:
        tokens = [token for token in phrase.split() if token]
        return any(character.isdigit() for character in phrase) or len(tokens) >= 2

    def _infer_open_entity_type(
        self,
        normalized_text: str,
        *,
        surface: str,
        full_text: str,
    ) -> str | None:
        canonical_ids = {
            alias.canonical_entity_id
            for alias in self.alias_registry.find_normalized_aliases(normalized_text)
        }
        if canonical_ids:
            entity_types = {
                self.alias_registry.entities[canonical_id].entity_type
                for canonical_id in canonical_ids
            }
            if len(entity_types) == 1:
                return next(iter(entity_types))

        entity_type_counts: dict[str, int] = defaultdict(int)
        for token in _significant_tokens(normalized_text):
            for alias in self._alias_token_index.get(token, []):
                entity = self.alias_registry.entities[alias.canonical_entity_id]
                entity_type_counts[entity.entity_type] += 1
        if entity_type_counts:
            for entity_type in _OPEN_TYPE_PRIORITY:
                if entity_type_counts.get(entity_type, 0) >= 2:
                    return entity_type

        compact_text = normalized_text.replace(" ", "")
        if any(character.isdigit() for character in compact_text):
            return "product"

        combined_text = f"{normalize_alias(surface)} || {normalize_alias(full_text)}"
        for entity_type in _OPEN_TYPE_PRIORITY:
            cues = _TYPE_CUES.get(entity_type, ())
            if any(cue in combined_text for cue in cues):
                return entity_type
        return None
