from __future__ import annotations
import importlib.metadata
import importlib.util
from smap.providers.base import EmbeddingProvider, EmbeddingPurpose, ProviderProvenance, SimilarityMatch
from smap.providers.cache import EmbeddingCacheStore
from smap.providers.errors import ProviderUnavailableError
from smap.providers.phobert_runtime import PhoBERTSemanticService, resolve_device

def _cosine_similarity(left, right):
    numerator = sum((a * b for a, b in zip(left, right, strict=True)))
    left_norm = sum((a * a for a in left)) ** 0.5
    right_norm = sum((b * b for b in right)) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))

class PhoBERTEmbeddingProvider(EmbeddingProvider):

    def __init__(self, *, model_id, device='auto', batch_size=16, max_length=256, cache_store=None, phrase_lexicon=None):
        if importlib.util.find_spec('transformers') is None:
            raise ProviderUnavailableError('transformers is unavailable for PhoBERT embeddings.')
        try:
            package_version = importlib.metadata.version('transformers')
        except importlib.metadata.PackageNotFoundError:
            package_version = 'unknown'
        self.model_id = model_id
        self.batch_size = max(batch_size, 1)
        self.max_length = max(max_length, 32)
        self.cache_store = cache_store
        self.device = resolve_device(device)
        self.runtime = PhoBERTSemanticService.shared(model_id=model_id, device=self.device, batch_size=self.batch_size, max_length=self.max_length)
        if phrase_lexicon:
            self.runtime.update_phrase_lexicon(list(phrase_lexicon))
        self.version = f'phobert-embedding:{model_id}'
        self._memory_cache: dict[tuple[str, str], tuple[float, ...]] = {}
        self.provenance = ProviderProvenance(provider_kind='embedding', provider_name='phobert_embedding', provider_version=package_version, model_id=model_id, device=self.runtime.actual_device(), run_metadata={'requested_device': device, 'resolved_device': self.runtime.actual_device(), 'batch_size': self.batch_size, 'max_length': self.max_length})

    def embed_texts(self, texts, *, purpose=EmbeddingPurpose.PASSAGE):
        if not texts:
            return []
        unique_texts = list(dict.fromkeys((self._prepare_text(text, purpose) for text in texts)))

        def memory_key(value):
            return (purpose.value, value)
        cached_vectors = {text: self._memory_cache[memory_key(text)] for text in unique_texts if memory_key(text) in self._memory_cache}
        missing = [text for text in unique_texts if text not in cached_vectors]
        store_hits = self.cache_store.load_many(model_id=self.model_id, purpose=purpose.value, texts=missing) if self.cache_store is not None and missing else {}
        for text, vector in store_hits.items():
            cached_vectors[text] = vector
            self._memory_cache[memory_key(text)] = vector
        missing = [text for text in unique_texts if text not in cached_vectors]
        if missing:
            encoded = self.runtime.encode_texts(missing, purpose=purpose, texts_are_prepared=True)
            if self.cache_store is not None:
                self.cache_store.store_many([{'model_id': self.model_id, 'purpose': purpose.value, 'text': text, 'vector': vector} for text, vector in zip(missing, encoded, strict=True)])
            for text, vector in zip(missing, encoded, strict=True):
                cached_vectors[text] = vector
                self._memory_cache[memory_key(text)] = vector
        prepared = [self._prepare_text(text, purpose) for text in texts]
        return [cached_vectors[text] for text in prepared]

    def rank_candidates(self, text, candidates, *, purpose=EmbeddingPurpose.LINKING, top_k=5):
        if not candidates:
            return []
        query_vector = self.embed_texts([text], purpose=purpose)[0]
        candidate_items = list(candidates.items())
        candidate_vectors = self.embed_texts([candidate_text for _, candidate_text in candidate_items], purpose=EmbeddingPurpose.PASSAGE)
        ranked = [SimilarityMatch(candidate_id=candidate_id, score=round(_cosine_similarity(query_vector, candidate_vector), 6), candidate_text=candidate_text) for (candidate_id, candidate_text), candidate_vector in zip(candidate_items, candidate_vectors, strict=True)]
        return sorted(ranked, key=lambda item: (-item.score, item.candidate_id))[:top_k]

    def best_match(self, text, candidates):
        ranked = self.rank_candidates(text, candidates, top_k=1)
        return ranked[0] if ranked else None

    def close(self):
        cache_store = getattr(self, 'cache_store', None)
        close_method = getattr(cache_store, 'close', None)
        if callable(close_method):
            close_method()

    def set_phrase_lexicon(self, phrases):
        self.runtime.update_phrase_lexicon(list(phrases))
        self._memory_cache.clear()

    def _prepare_text(self, text, purpose):
        return self.runtime.prepare_text(text, purpose=purpose)
