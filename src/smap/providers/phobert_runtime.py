from __future__ import annotations
import threading
import unicodedata
from contextlib import suppress
from typing import Any
from smap.providers.base import EmbeddingPurpose

def resolve_device(requested_device):
    if requested_device != 'auto':
        return requested_device
    try:
        import torch
    except ImportError:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return 'mps'
    return 'cpu'

class PhoBERTSemanticService:
    _CACHE: dict[tuple[str, str, int, int], PhoBERTSemanticService] = {}
    _LOCK = threading.Lock()

    def __init__(self, *, model_id, device, batch_size=16, max_length=256):
        self.model_id = model_id
        self.device = resolve_device(device)
        self.batch_size = max(batch_size, 1)
        self.max_length = max(max_length, 32)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._prepare_cache: dict[tuple[str, str], str] = {}
        self._phrase_token_index: dict[str, list[tuple[str, ...]]] = {}
        self._phrase_lexicon: tuple[str, ...] = ()

    @classmethod
    def shared(cls, *, model_id, device, batch_size=16, max_length=256):
        cache_key = (model_id, resolve_device(device), max(batch_size, 1), max(max_length, 32))
        with cls._LOCK:
            runtime = cls._CACHE.get(cache_key)
            if runtime is None:
                runtime = cls(model_id=model_id, device=device, batch_size=batch_size, max_length=max_length)
                cls._CACHE[cache_key] = runtime
        return runtime

    def update_phrase_lexicon(self, phrases):
        normalized_phrases = tuple(sorted({self._normalize_surface(phrase) for phrase in phrases if len(self._normalize_surface(phrase).split()) >= 2 or any((character.isdigit() for character in self._normalize_surface(phrase)))}, key=lambda item: (-len(item.split()), -len(item), item)))
        if normalized_phrases == self._phrase_lexicon:
            return
        phrase_token_index: dict[str, list[tuple[str, ...]]] = {}
        for phrase in normalized_phrases:
            tokens = tuple(phrase.split())
            if not tokens:
                continue
            phrase_token_index.setdefault(tokens[0], []).append(tokens)
        for token, entries in phrase_token_index.items():
            phrase_token_index[token] = sorted(entries, key=lambda item: (-len(item), item))
        self._phrase_lexicon = normalized_phrases
        self._phrase_token_index = phrase_token_index
        self._prepare_cache.clear()

    def prepare_text(self, text, *, purpose=EmbeddingPurpose.PASSAGE):
        cache_key = (purpose.value, text)
        cached = self._prepare_cache.get(cache_key)
        if cached is not None:
            return cached
        normalized = self._normalize_surface(text)
        segmented = self._apply_phrase_segmentation(normalized)
        if not segmented:
            segmented = '<empty>'
        prefix = {EmbeddingPurpose.QUERY: 'query:', EmbeddingPurpose.LINKING: 'entity:', EmbeddingPurpose.CLUSTERING: 'topic:', EmbeddingPurpose.PASSAGE: ''}[purpose]
        prepared = f'{prefix} {segmented}'.strip()
        self._prepare_cache[cache_key] = prepared
        return prepared

    def encode_texts(self, texts, *, purpose=EmbeddingPurpose.PASSAGE, texts_are_prepared=False):
        if not texts:
            return []
        torch, tokenizer, model = self._runtime()
        prepared_texts = texts if texts_are_prepared else [self.prepare_text(text, purpose=purpose) for text in texts]
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(prepared_texts), self.batch_size):
            batch = prepared_texts[start:start + self.batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
                hidden = outputs.last_hidden_state
                attention_mask = encoded['attention_mask'].unsqueeze(-1)
                pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                batch_vectors = normalized.detach().cpu().tolist()
            vectors.extend((tuple((float(value) for value in vector)) for vector in batch_vectors))
        return vectors

    def actual_device(self):
        _, _, model = self._runtime()
        with suppress(Exception):
            parameter = next(model.parameters())
            return str(parameter.device)
        return self.device

    def _apply_phrase_segmentation(self, normalized):
        if not normalized:
            return ''
        tokens = normalized.split()
        if not tokens or not self._phrase_token_index:
            return ' '.join(tokens)
        segmented: list[str] = []
        index = 0
        token_count = len(tokens)
        while index < token_count:
            first = tokens[index]
            match: tuple[str, ...] | None = None
            for candidate in self._phrase_token_index.get(first, []):
                size = len(candidate)
                if tokens[index:index + size] == list(candidate):
                    match = candidate
                    break
            if match is not None:
                segmented.append('_'.join(match))
                index += len(match)
            else:
                segmented.append(tokens[index])
                index += 1
        return ' '.join(segmented)

    def _normalize_surface(self, text):
        normalized = unicodedata.normalize('NFKC', text)
        normalized = normalized.replace('_', ' ').replace('-', ' ')
        return ' '.join(normalized.split()).casefold()

    def _runtime(self):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('PhoBERT runtime requires `transformers` and `torch`.') from exc
        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()
        return (torch, self._tokenizer, self._model)
PhoBERTRuntime = PhoBERTSemanticService
