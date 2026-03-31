from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

from smap.providers.base import (
    LanguageIdentificationResult,
    LanguageIdProvider,
    ProviderProvenance,
)
from smap.providers.errors import ProviderUnavailableError
from smap.providers.lid_heuristic import (
    best_language_candidate,
    canonicalize_language_label,
    heuristic_language_candidates,
)


class FastTextLanguageIdProvider(LanguageIdProvider):
    def __init__(
        self,
        *,
        model_path: Path,
        mixed_confidence_threshold: float = 0.55,
        mixed_gap_threshold: float = 0.12,
        script_override_enabled: bool = True,
    ) -> None:
        try:
            import fasttext  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailableError("fastText language ID backend is unavailable.") from exc
        if not model_path.exists():
            raise ProviderUnavailableError(f"fastText language ID model is missing: {model_path}")
        self._fasttext = fasttext
        self.model_path = model_path
        self.mixed_confidence_threshold = mixed_confidence_threshold
        self.mixed_gap_threshold = mixed_gap_threshold
        self.script_override_enabled = script_override_enabled
        self.version = "fasttext-lid-v1"
        try:
            package_version = importlib.metadata.version("fasttext")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - optional dependency
            package_version = "unknown"
        self.provenance = ProviderProvenance(
            provider_kind="language_id",
            provider_name="fasttext_lid",
            provider_version=package_version,
            model_id=model_path.name,
            device="cpu",
            run_metadata={
                "model_path": str(model_path),
                "mixed_confidence_threshold": mixed_confidence_threshold,
                "mixed_gap_threshold": mixed_gap_threshold,
                "script_override_enabled": script_override_enabled,
            },
        )
        self._model: Any | None = None

    def detect(self, text: str) -> LanguageIdentificationResult:
        if not text.strip():
            return LanguageIdentificationResult(
                language="unknown",
                confidence=0.0,
                provider_provenance=self.provenance,
                metadata={"reason": "empty_text"},
            )
        model = self._model_instance()
        labels, probabilities = self._predict_topk(model, text.replace("\n", " "), k=2)
        candidates = {
            canonicalize_language_label(label): round(float(probability), 4)
            for label, probability in zip(labels, probabilities, strict=True)
        }
        heuristic = heuristic_language_candidates(text)
        for language, score in heuristic.items():
            if language == "unknown":
                continue
            candidates[language] = max(candidates.get(language, 0.0), round(score * 0.72, 4))
        language, confidence, filtered = best_language_candidate(
            candidates,
            text=text,
            mixed_confidence_threshold=self.mixed_confidence_threshold,
            mixed_gap_threshold=self.mixed_gap_threshold,
            script_override_enabled=self.script_override_enabled,
        )
        top_candidates = sorted(filtered.items(), key=lambda item: (-item[1], item[0]))[:3]
        return LanguageIdentificationResult(
            language=language,
            confidence=confidence,
            provider_provenance=self.provenance,
            metadata={
                "top_candidates": ", ".join(f"{item[0]}:{item[1]:.3f}" for item in top_candidates),
            },
        )

    def _model_instance(self) -> Any:
        if self._model is None:
            try:
                self._model = self._fasttext.load_model(str(self.model_path))
            except Exception as exc:  # pragma: no cover - local model state
                raise ProviderUnavailableError(
                    f"Unable to load fastText language ID model from `{self.model_path}`."
                ) from exc
        return self._model

    def _predict_topk(
        self,
        model: Any,
        text: str,
        *,
        k: int,
    ) -> tuple[list[str], list[float]]:
        # fasttext's Python wrapper currently breaks on NumPy 2 because it calls
        # np.array(..., copy=False). The lower-level pybind path remains stable,
        # so prefer that and only fall back to the wrapper when necessary.
        raw_model = getattr(model, "f", None)
        if raw_model is not None and hasattr(raw_model, "predict"):
            predictions = raw_model.predict(f"{text}\n", k, 0.0, "strict")
            labels: list[str] = []
            probabilities: list[float] = []
            for prediction in predictions:
                if not isinstance(prediction, tuple) or len(prediction) < 2:
                    continue
                first, second = prediction[0], prediction[1]
                if isinstance(first, str):
                    label = first
                    probability = float(second)
                else:
                    label = str(second)
                    probability = float(first)
                labels.append(label)
                probabilities.append(probability)
            if labels:
                return labels, probabilities
        labels, probabilities = model.predict(text, k=k)
        return [str(label) for label in labels], [float(probability) for probability in probabilities]
