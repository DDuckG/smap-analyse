from __future__ import annotations

import threading
import unicodedata
from pathlib import Path
from typing import Any

from smap.providers.base import EmbeddingPurpose


def resolve_device(requested_device: str) -> str:
    if requested_device not in {"auto", "cpu"}:
        raise ValueError("PhoBERT ONNX runtime only supports CPU execution.")
    return "cpu"


class PhoBERTSemanticService:
    _CACHE: dict[tuple[str, int, int, str, int | None, int | None], PhoBERTSemanticService] = {}
    _LOCK = threading.Lock()

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        batch_size: int = 24,
        max_length: int = 256,
        backend: str = "onnx",
        onnx_dir: Path | None = None,
        onnx_intra_op_threads: int | None = None,
        onnx_inter_op_threads: int | None = None,
    ) -> None:
        if backend != "onnx":
            raise ValueError("Only the ONNX embedding backend is supported.")
        self.model_id = model_id
        self.backend = backend
        self.requested_device = device
        self.device = resolve_device(device)
        self.batch_size = max(batch_size, 1)
        self.max_length = max(max_length, 32)
        self.onnx_dir = onnx_dir
        self.onnx_intra_op_threads = onnx_intra_op_threads
        self.onnx_inter_op_threads = onnx_inter_op_threads
        self._onnx_export_path = (
            self.onnx_dir / _safe_model_cache_dir(self.model_id) if self.onnx_dir is not None else None
        )
        self._onnx_model_source = "uninitialized"
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._prepare_cache: dict[tuple[str, str], str] = {}
        self._phrase_token_index: dict[str, list[tuple[str, ...]]] = {}
        self._phrase_lexicon: tuple[str, ...] = ()
        self._encode_invocations = 0
        self._encoded_text_count = 0
        self._executed_batch_count = 0
        self._last_batch_sizes: tuple[int, ...] = ()

    @classmethod
    def shared(
        cls,
        *,
        model_id: str,
        device: str,
        batch_size: int = 24,
        max_length: int = 256,
        backend: str = "onnx",
        onnx_dir: Path | None = None,
        onnx_intra_op_threads: int | None = None,
        onnx_inter_op_threads: int | None = None,
    ) -> PhoBERTSemanticService:
        cache_key = (
            model_id,
            max(batch_size, 1),
            max(max_length, 32),
            str(onnx_dir or ""),
            onnx_intra_op_threads,
            onnx_inter_op_threads,
        )
        with cls._LOCK:
            runtime = cls._CACHE.get(cache_key)
            if runtime is None:
                runtime = cls(
                    model_id=model_id,
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length,
                    backend=backend,
                    onnx_dir=onnx_dir,
                    onnx_intra_op_threads=onnx_intra_op_threads,
                    onnx_inter_op_threads=onnx_inter_op_threads,
                )
                cls._CACHE[cache_key] = runtime
        return runtime

    def update_phrase_lexicon(self, phrases: list[str] | tuple[str, ...]) -> None:
        normalized_phrases = tuple(
            sorted(
                {
                    self._normalize_surface(phrase)
                    for phrase in phrases
                    if len(self._normalize_surface(phrase).split()) >= 2
                    or any(character.isdigit() for character in self._normalize_surface(phrase))
                },
                key=lambda item: (-len(item.split()), -len(item), item),
            )
        )
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

    def prepare_text(self, text: str, *, purpose: EmbeddingPurpose = EmbeddingPurpose.PASSAGE) -> str:
        cache_key = (purpose.value, text)
        cached = self._prepare_cache.get(cache_key)
        if cached is not None:
            return cached
        normalized = self._normalize_surface(text)
        segmented = self._apply_phrase_segmentation(normalized)
        if not segmented:
            segmented = "<empty>"
        prefix = {
            EmbeddingPurpose.QUERY: "query:",
            EmbeddingPurpose.LINKING: "entity:",
            EmbeddingPurpose.CLUSTERING: "topic:",
            EmbeddingPurpose.PASSAGE: "",
        }[purpose]
        prepared = f"{prefix} {segmented}".strip()
        self._prepare_cache[cache_key] = prepared
        return prepared

    def encode_texts(
        self,
        texts: list[str],
        *,
        purpose: EmbeddingPurpose = EmbeddingPurpose.PASSAGE,
        texts_are_prepared: bool = False,
    ) -> list[tuple[float, ...]]:
        if not texts:
            return []
        return self._encode_texts_onnx(texts, purpose=purpose, texts_are_prepared=texts_are_prepared)

    def actual_device(self) -> str:
        return "cpu"

    def provenance_metadata(self) -> dict[str, str | int | bool | None]:
        return {
            "backend": "onnx",
            "requested_device": self.requested_device,
            "resolved_device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "onnx_export_dir": str(self._onnx_export_path) if self._onnx_export_path is not None else None,
            "onnx_export_available": self._onnx_export_available(),
            "onnx_model_source": self._onnx_model_source,
            "onnx_intra_op_threads": self.onnx_intra_op_threads,
            "onnx_inter_op_threads": self.onnx_inter_op_threads,
        }

    def debug_snapshot(self) -> dict[str, object]:
        return {
            **self.provenance_metadata(),
            "prepare_cache_entries": len(self._prepare_cache),
            "phrase_lexicon_size": len(self._phrase_lexicon),
            "encode_invocations": self._encode_invocations,
            "encoded_texts": self._encoded_text_count,
            "executed_batch_count": self._executed_batch_count,
            "last_batch_sizes": list(self._last_batch_sizes),
        }

    def _apply_phrase_segmentation(self, normalized: str) -> str:
        if not normalized:
            return ""
        tokens = normalized.split()
        if not tokens or not self._phrase_token_index:
            return " ".join(tokens)
        segmented: list[str] = []
        index = 0
        token_count = len(tokens)
        while index < token_count:
            first = tokens[index]
            match: tuple[str, ...] | None = None
            for candidate in self._phrase_token_index.get(first, []):
                size = len(candidate)
                if tokens[index : index + size] == list(candidate):
                    match = candidate
                    break
            if match is not None:
                segmented.append("_".join(match))
                index += len(match)
            else:
                segmented.append(tokens[index])
                index += 1
        return " ".join(segmented)

    def _normalize_surface(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("_", " ").replace("-", " ")
        return " ".join(normalized.split()).casefold()

    def _record_encode_stats(self, *, text_count: int, batch_sizes: list[int]) -> None:
        self._encode_invocations += 1
        self._encoded_text_count += text_count
        self._executed_batch_count += len(batch_sizes)
        self._last_batch_sizes = tuple(batch_sizes[-8:])

    def _onnx_export_available(self) -> bool:
        export_path = self._onnx_export_path
        return export_path is not None and export_path.exists() and any(export_path.glob("*.onnx"))

    def _encode_texts_onnx(
        self,
        texts: list[str],
        *,
        purpose: EmbeddingPurpose = EmbeddingPurpose.PASSAGE,
        texts_are_prepared: bool = False,
    ) -> list[tuple[float, ...]]:
        import numpy as np

        tokenizer, model = self._onnx_runtime()
        prepared_texts = texts if texts_are_prepared else [self.prepare_text(text, purpose=purpose) for text in texts]
        vectors: list[tuple[float, ...]] = []
        batch_sizes: list[int] = []
        for start in range(0, len(prepared_texts), self.batch_size):
            batch = prepared_texts[start : start + self.batch_size]
            batch_sizes.append(len(batch))
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            encoded_inputs = {
                key: value.astype("int64")
                for key, value in encoded.items()
                if key in getattr(model, "input_names", set(encoded))
            }
            outputs = model(**encoded_inputs, return_dict=True)
            hidden = outputs.last_hidden_state
            attention_mask = encoded_inputs["attention_mask"].astype("float32")[..., np.newaxis]
            pooled = (hidden * attention_mask).sum(axis=1) / np.clip(attention_mask.sum(axis=1), 1.0, None)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalized = pooled / np.clip(norms, 1e-12, None)
            vectors.extend(tuple(float(value) for value in vector.tolist()) for vector in normalized)
        self._record_encode_stats(text_count=len(prepared_texts), batch_sizes=batch_sizes)
        return vectors

    def _onnx_runtime(self) -> tuple[Any, Any]:
        try:
            import onnxruntime
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PhoBERT ONNX runtime requires `onnxruntime`, `optimum-onnx[onnxruntime]`, and `transformers`."
            ) from exc

        if self._tokenizer is None or self._model is None:
            export_dir = self._onnx_export_path
            if export_dir is not None:
                export_dir.mkdir(parents=True, exist_ok=True)
            model_dir = export_dir if self._onnx_export_available() else None
            tokenizer_source = (
                model_dir
                if model_dir is not None and (model_dir / "tokenizer_config.json").exists()
                else self.model_id
            )
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
            session_options = onnxruntime.SessionOptions()
            if self.onnx_intra_op_threads is not None:
                session_options.intra_op_num_threads = self.onnx_intra_op_threads
            if self.onnx_inter_op_threads is not None:
                session_options.inter_op_num_threads = self.onnx_inter_op_threads
            if model_dir is None:
                try:
                    self._model = ORTModelForFeatureExtraction.from_pretrained(
                        self.model_id,
                        export=True,
                        provider="CPUExecutionProvider",
                        session_options=session_options,
                    )
                except Exception as exc:  # pragma: no cover - environment-dependent
                    export_target = (
                        str(export_dir)
                        if export_dir is not None
                        else "the configured ONNX model directory"
                    )
                    raise RuntimeError(
                        "PhoBERT ONNX runtime could not initialize because no exported model was found at "
                        f"{export_target} and automatic export failed. Stage a pre-exported ONNX model under "
                        "that directory or provide the dependencies required by Optimum export."
                    ) from exc
                if export_dir is not None:
                    self._model.save_pretrained(export_dir)
                    self._tokenizer.save_pretrained(export_dir)
                self._onnx_model_source = "fresh_export"
            else:
                self._model = ORTModelForFeatureExtraction.from_pretrained(
                    model_dir,
                    provider="CPUExecutionProvider",
                    session_options=session_options,
                    local_files_only=True,
                )
                self._onnx_model_source = "cached_export"
        return self._tokenizer, self._model


def _safe_model_cache_dir(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__").replace(":", "_")


PhoBERTRuntime = PhoBERTSemanticService
