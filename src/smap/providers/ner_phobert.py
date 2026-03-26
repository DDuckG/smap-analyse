from __future__ import annotations
import importlib.metadata
import re
from collections import defaultdict
from smap.canonicalization.alias import AliasRegistry, boundary_spans, normalize_alias
from smap.canonicalization.models import EntityAlias
from smap.providers.base import EmbeddingProvider, EmbeddingPurpose, NERProvider, ProviderProvenance, RecognizedEntitySpan
from smap.providers.errors import ProviderUnavailableError
_MODEL_CODE_RE = re.compile('\\b[a-z]{1,10}[- ]?\\d{1,4}[a-z0-9-]{0,4}\\b', re.IGNORECASE)
_ENTITY_TOKEN_RE = re.compile('[A-Za-z0-9][A-Za-z0-9-]{1,}')
_CAMEL_TITLE_RE = re.compile('\\b[A-ZÀ-Ỹ][\\w-]+(?:\\s+[A-ZÀ-Ỹ][\\w-]+){0,3}\\b')
_GENERIC_SURFACES = {'brand', 'company', 'customer service', 'factory', 'founder', 'item', 'mall', 'market', 'model', 'organization', 'official mall', 'plant', 'product', 'service', 'showroom', 'store'}
_TYPE_PROTOTYPES = {'brand': 'thuong hieu brand hang xe cong ty brand name', 'product': 'san pham product model dong xe phien ban variant mau xe', 'organization': 'to chuc organization tap doan company doanh nghiep', 'person': 'nhan vat person founder chairman ceo nguoi noi tieng', 'location': 'dia diem location thanh pho quoc gia thi truong market', 'facility': 'nha may factory xưởng plant facility production site', 'retailer': 'dai ly showroom dealer retailer cua hang store', 'concept': 'khai niem concept chu de generic'}

class PhoBERTNERProvider(NERProvider):

    def __init__(self, *, alias_registry, embedding_provider, model_id, min_similarity=0.58):
        if embedding_provider.provenance.provider_name != 'phobert_embedding':
            raise ProviderUnavailableError('PhoBERT NER requires a PhoBERT embedding provider.')
        try:
            package_version = importlib.metadata.version('transformers')
        except importlib.metadata.PackageNotFoundError:
            package_version = 'unknown'
        self.alias_registry = alias_registry
        self.embedding_provider = embedding_provider
        self.model_id = model_id
        self.min_similarity = min_similarity
        self.version = f'phobert-ner:{model_id}'
        self.provenance = ProviderProvenance(provider_kind='ner', provider_name='phobert_ner', provider_version=package_version, model_id=model_id, device=embedding_provider.provenance.device, run_metadata={'min_similarity': min_similarity})
        self._alias_token_index = self._build_alias_token_index()
        self._entity_types = self._build_entity_types()

    def extract(self, text, *, mention_id, source_uap_id, label_inventory=None):
        del mention_id, source_uap_id
        allowed_types = set(label_inventory or self._entity_types)
        spans: list[RecognizedEntitySpan] = []
        seen: set[tuple[int, int, str]] = set()
        for start, end, surface in self._span_proposals(text):
            normalized = normalize_alias(surface)
            if not normalized or normalized in _GENERIC_SURFACES:
                continue
            top_alias = self._top_alias_match(surface, normalized)
            top_type = self._top_type_match(surface, normalized, allowed_types)
            if top_alias is not None and top_alias[2] >= max(self.min_similarity, 0.66):
                entity = self.alias_registry.entities[top_alias[0]]
                key = (start, end, entity.entity_type)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(RecognizedEntitySpan(start=start, end=end, text=surface, normalized_text=normalized, label=entity.entity_type, entity_type_hint=entity.entity_type, confidence=round(min(max(top_alias[2], 0.72), 0.99), 3), provider_provenance=self.provenance, metadata={'source': 'phobert_ner', 'matched_entity_id': top_alias[0]}))
                continue
            if top_type is None:
                continue
            entity_type, score = top_type
            if not self._span_is_worthy(surface, normalized, entity_type):
                continue
            specificity_bonus = 0.08 if any((character.isdigit() for character in normalized)) else 0.0
            final_score = score + specificity_bonus
            if final_score < self.min_similarity:
                continue
            key = (start, end, entity_type)
            if key in seen:
                continue
            seen.add(key)
            spans.append(RecognizedEntitySpan(start=start, end=end, text=surface, normalized_text=normalized, label=entity_type, entity_type_hint=entity_type, confidence=round(min(final_score, 0.92), 3), provider_provenance=self.provenance, metadata={'source': 'phobert_ner'}))
        return sorted(spans, key=lambda item: (item.start, item.end, item.label))

    def _span_is_worthy(self, surface, normalized, entity_type):
        tokens = [token for token in normalized.split() if token]
        compact = normalized.replace(' ', '')
        if not compact or compact in _GENERIC_SURFACES:
            return False
        if any((character.isdigit() for character in compact)):
            return True
        if normalized in self.alias_registry.aliases_by_normalized:
            return True
        if entity_type in {'brand', 'organization', 'retailer'}:
            return len(compact) >= 4 and (any((character.isupper() for character in surface)) or len(tokens) >= 2 or compact in self._alias_token_index)
        if entity_type == 'product':
            return len(compact) >= 4 and (len(tokens) >= 2 or '-' in surface or compact in self._alias_token_index)
        if entity_type in {'facility', 'location'}:
            return len(tokens) >= 2
        if entity_type == 'person':
            return len(tokens) >= 2 and all((token[:1].isupper() for token in surface.split()[:2] if token))
        if entity_type == 'concept':
            return len(tokens) >= 2 and compact not in _GENERIC_SURFACES
        return len(compact) >= 4

    def _span_proposals(self, text):
        proposals: list[tuple[int, int, str]] = []
        lowered = text.casefold()
        for alias in self.alias_registry.all_aliases():
            for start, end in boundary_spans(lowered, alias.alias.casefold()):
                proposals.append((start, end, text[start:end]))
        for match in _MODEL_CODE_RE.finditer(text):
            proposals.append((match.start(), match.end(), match.group(0)))
        for match in _CAMEL_TITLE_RE.finditer(text):
            surface = match.group(0)
            normalized = normalize_alias(surface)
            if not self._supports_title_surface(surface, normalized):
                continue
            proposals.append((match.start(), match.end(), surface))
        token_matches = list(_ENTITY_TOKEN_RE.finditer(text))
        for index, match in enumerate(token_matches):
            token = match.group(0)
            normalized = normalize_alias(token)
            if len(normalized) < 2:
                continue
            single_token_exact_alias = bool(self.alias_registry.find_normalized_aliases(normalized))
            if not any((character.isdigit() for character in normalized)) and (not single_token_exact_alias):
                continue
            if not self._proposal_is_business_worthy(token, normalized, exact_alias=single_token_exact_alias):
                continue
            if not single_token_exact_alias and self._covered_by_longer_proposal(match.start(), match.end(), proposals):
                continue
            proposals.append((match.start(), match.end(), token))
            if index + 1 < len(token_matches):
                second = token_matches[index + 1]
                combined = f'{token} {second.group(0)}'
                if self._supports_combined_surface(combined) and (not self._covered_by_longer_proposal(match.start(), second.end(), proposals)):
                    proposals.append((match.start(), second.end(), combined))
            if index + 2 < len(token_matches) and any((character.isdigit() for character in normalized)):
                third = token_matches[index + 2]
                combined = f'{token} {token_matches[index + 1].group(0)} {third.group(0)}'
                if self._supports_combined_surface(combined) and (not self._covered_by_longer_proposal(match.start(), third.end(), proposals)):
                    proposals.append((match.start(), third.end(), combined))
        deduped: dict[tuple[int, int, str], tuple[int, int, str]] = {}
        for start, end, surface in proposals:
            cleaned = surface.strip()
            if len(cleaned) < 2:
                continue
            deduped[start, end, normalize_alias(cleaned)] = (start, end, cleaned)
        return sorted(deduped.values(), key=lambda item: (item[0], item[1] - item[0]))

    def _supports_title_surface(self, surface, normalized):
        if normalized in _GENERIC_SURFACES:
            return False
        tokens = [token for token in normalized.split() if token]
        if any((character.isdigit() for character in normalized)):
            return True
        if len(tokens) >= 2 and any((character.isupper() for character in surface)):
            return True
        return normalized in self._alias_token_index

    def _supports_combined_surface(self, surface):
        normalized = normalize_alias(surface)
        if normalized in _GENERIC_SURFACES:
            return False
        tokens = [token for token in normalized.split() if token]
        if len(tokens) < 2:
            return False
        if any((character.isdigit() for character in normalized)):
            return True
        return sum((1 for token in tokens if token in self._alias_token_index)) >= 1

    def _proposal_is_business_worthy(self, surface, normalized, *, exact_alias):
        if normalized in _GENERIC_SURFACES:
            return False
        if any((character.isdigit() for character in normalized)):
            return True
        if exact_alias:
            return len(normalized) >= 4 or any((character.isupper() for character in surface))
        return len(normalized) >= 5 and any((character.isupper() for character in surface))

    def _covered_by_longer_proposal(self, start, end, proposals):
        for existing_start, existing_end, _ in proposals:
            if existing_start <= start and end <= existing_end and (existing_end - existing_start > end - start):
                return True
        return False

    def _top_alias_match(self, surface, normalized):
        exact = self.alias_registry.find_exact_aliases(surface)
        if exact:
            alias = exact[0]
            return (alias.canonical_entity_id, alias.alias, 0.99)
        normalized_hits = self.alias_registry.find_normalized_aliases(normalized)
        if normalized_hits:
            alias = normalized_hits[0]
            return (alias.canonical_entity_id, alias.alias, 0.96)
        shortlist = self._shortlist_aliases(normalized)
        if not shortlist:
            return None
        ranked = self.embedding_provider.rank_candidates(normalized, {alias.alias_id: alias.alias for alias in shortlist}, purpose=EmbeddingPurpose.LINKING, top_k=3)
        if not ranked:
            return None
        top = ranked[0]
        matched_alias = next((item for item in shortlist if item.alias_id == top.candidate_id), None)
        if matched_alias is None:
            return None
        return (matched_alias.canonical_entity_id, matched_alias.alias, top.score)

    def _top_type_match(self, surface, normalized, allowed_types):
        type_candidates = {entity_type: prototype for entity_type, prototype in _TYPE_PROTOTYPES.items() if entity_type in allowed_types}
        if not type_candidates:
            return None
        query = surface if any((character.isupper() for character in surface)) else normalized
        ranked = self.embedding_provider.rank_candidates(query, type_candidates, purpose=EmbeddingPurpose.LINKING, top_k=1)
        if not ranked:
            return None
        return (ranked[0].candidate_id, ranked[0].score)

    def _shortlist_aliases(self, normalized):
        tokens = {token for token in normalized.split() if len(token) >= 2 or any((character.isdigit() for character in token))}
        if not tokens and normalized:
            tokens = {normalized}
        alias_ids = {alias_id for token in tokens for alias_id in self._alias_token_index.get(token, set())}
        if not alias_ids and any((character.isdigit() for character in normalized)):
            alias_ids = {alias.alias_id for alias in self.alias_registry.all_aliases() if any((character.isdigit() for character in alias.normalized_alias))}
        shortlist = [alias for alias in self.alias_registry.all_aliases() if alias.alias_id in alias_ids]
        return shortlist[:24]

    def _build_alias_token_index(self):
        token_index: dict[str, set[str]] = defaultdict(set)
        for alias in self.alias_registry.all_aliases():
            for token in alias.normalized_alias.split():
                token_index[token].add(alias.alias_id)
            compact = alias.normalized_alias.replace(' ', '')
            if compact:
                token_index[compact].add(alias.alias_id)
        return token_index

    def _build_entity_types(self):
        return sorted({entity.entity_type for entity in self.alias_registry.entities.values() if entity.entity_type in _TYPE_PROTOTYPES})
