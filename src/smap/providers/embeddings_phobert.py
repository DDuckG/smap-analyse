from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

from smap.providers.base import (
    EmbeddingProvider,
    EmbeddingPurpose,
    ProviderProvenance,
    SimilarityMatch,
)
from smap.providers.cache import EmbeddingCacheStore
from smap.providers.errors import ProviderUnavailableError
from smap.providers.phobert_runtime import PhoBERTSemanticService, resolve_device


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


class PhoBERTEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
        batch_size: int = 24,
        max_length: int = 256,
        cache_store: EmbeddingCacheStore | None = None,
        phrase_lexicon: list[str] | tuple[str, ...] | None = None,
        runtime_backend: str = "onnx",
        onnx_dir: Path | None = None,
        onnx_intra_op_threads: int | None = None,
        onnx_inter_op_threads: int | None = None,
    ) -> None:
        if runtime_backend != "onnx":
            raise ProviderUnavailableError("Only ONNX CPU embeddings are supported.")
        if importlib.util.find_spec("transformers") is None:  # pragma: no cover - optional dependency
            raise ProviderUnavailableError("transformers is unavailable for PhoBERT embeddings.")
        try:
            package_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - optional dependency
            package_version = "unknown"
        self.model_id = model_id
        self.batch_size = max(batch_size, 1)
        self.max_length = max(max_length, 32)
        self.cache_store = cache_store
        self.runtime_backend = "onnx"
        self.device = resolve_device(device)
        self.runtime = PhoBERTSemanticService.shared(
            model_id=model_id,
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length,
            backend=self.runtime_backend,
            onnx_dir=onnx_dir,
            onnx_intra_op_threads=onnx_intra_op_threads,
            onnx_inter_op_threads=onnx_inter_op_threads,
        )
        if phrase_lexicon:
            self.runtime.update_phrase_lexicon(list(phrase_lexicon))
        self.version = f"phobert-embedding:{model_id}:onnx"
        self.cache_model_id = f"{model_id}|backend=onnx|max_length={self.max_length}"
        self._memory_cache: dict[tuple[str, str], tuple[float, ...]] = {}
        self._cache_stats = {
            "requests": 0,
            "unique_prepared_texts": 0,
            "memory_hits": 0,
            "store_hits": 0,
            "misses": 0,
            "encodes": 0,
            "stored": 0,
        }
        runtime_metadata = self.runtime.provenance_metadata()
        self.provenance = ProviderProvenance(
            provider_kind="embedding",
            provider_name="phobert_embedding",
            provider_version=package_version,
            model_id=model_id,
            device=self.runtime.actual_device(),
            run_metadata=runtime_metadata,
        )

    def embed_texts(
        self,
        texts: list[str],
        *,
        purpose: EmbeddingPurpose = EmbeddingPurpose.PASSAGE,
    ) -> list[tuple[float, ...]]:
        if not texts:
            return []
        prepared_texts = [self._prepare_text(text, purpose) for text in texts]
        unique_texts = list(dict.fromkeys(prepared_texts))
        self._cache_stats["requests"] += len(texts)
        self._cache_stats["unique_prepared_texts"] += len(unique_texts)

        def memory_key(value: str) -> tuple[str, str]:
            return purpose.value, value

        cached_vectors = {
            text: self._memory_cache[memory_key(text)]
            for text in unique_texts
            if memory_key(text) in self._memory_cache
        }
        self._cache_stats["memory_hits"] += len(cached_vectors)
        missing = [text for text in unique_texts if text not in cached_vectors]
        store_hits = (
            self.cache_store.load_many(model_id=self.cache_model_id, purpose=purpose.value, texts=missing)
            if self.cache_store is not None and missing
            else {}
        )
        for text, vector in store_hits.items():
            cached_vectors[text] = vector
            self._memory_cache[memory_key(text)] = vector
        self._cache_stats["store_hits"] += len(store_hits)
        missing = [text for text in unique_texts if text not in cached_vectors]
        if missing:
            self._cache_stats["misses"] += len(missing)
            self._cache_stats["encodes"] += 1
            encoded = self.runtime.encode_texts(missing, purpose=purpose, texts_are_prepared=True)
            if self.cache_store is not None:
                self.cache_store.store_many(
                    [
                        {
                            "model_id": self.cache_model_id,
                            "purpose": purpose.value,
                            "text": text,
                            "vector": vector,
                        }
                        for text, vector in zip(missing, encoded, strict=True)
                    ]
                )
                self._cache_stats["stored"] += len(missing)
            for text, vector in zip(missing, encoded, strict=True):
                cached_vectors[text] = vector
                self._memory_cache[memory_key(text)] = vector
        return [cached_vectors[text] for text in prepared_texts]

    def rank_candidates(
        self,
        text: str,
        candidates: dict[str, str],
        *,
        purpose: EmbeddingPurpose = EmbeddingPurpose.LINKING,
        top_k: int = 5,
    ) -> list[SimilarityMatch]:
        if not candidates:
            return []
        query_vector = self.embed_texts([text], purpose=purpose)[0]
        candidate_items = list(candidates.items())
        candidate_vectors = self.embed_texts(
            [candidate_text for _, candidate_text in candidate_items],
            purpose=EmbeddingPurpose.PASSAGE,
        )
        ranked = [
            SimilarityMatch(
                candidate_id=candidate_id,
                score=round(_cosine_similarity(query_vector, candidate_vector), 6),
                candidate_text=candidate_text,
            )
            for (candidate_id, candidate_text), candidate_vector in zip(
                candidate_items,
                candidate_vectors,
                strict=True,
            )
        ]
        return sorted(ranked, key=lambda item: (-item.score, item.candidate_id))[:top_k]

    def best_match(self, text: str, candidates: dict[str, str]) -> SimilarityMatch | None:
        ranked = self.rank_candidates(text, candidates, top_k=1)
        return ranked[0] if ranked else None

    def close(self) -> None:
        cache_store = getattr(self, "cache_store", None)
        close_method = getattr(cache_store, "close", None)
        if callable(close_method):
            close_method()

    def set_phrase_lexicon(self, phrases: list[str] | tuple[str, ...]) -> None:
        self.runtime.update_phrase_lexicon(list(phrases))
        self._memory_cache.clear()

    def _prepare_text(self, text: str, purpose: EmbeddingPurpose) -> str:
        return self.runtime.prepare_text(text, purpose=purpose)

    def cache_stats_snapshot(self) -> dict[str, int]:
        return {
            **self._cache_stats,
            "memory_entries": len(self._memory_cache),
        }

    def runtime_snapshot(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "cache_model_id": self.cache_model_id,
            "cache_store_path": str(self.cache_store.db_path) if self.cache_store is not None else None,
            "provider_cache": self.cache_stats_snapshot(),
            "runtime": self.runtime.debug_snapshot(),
        }
