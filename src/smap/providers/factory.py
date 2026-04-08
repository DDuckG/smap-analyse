from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from smap.canonicalization.alias import AliasRegistry
from smap.core.settings import Settings
from smap.ontology.models import OntologyRegistry
from smap.ontology.prototypes import collect_phrase_lexicon
from smap.ontology.runtime import load_runtime_ontology
from smap.providers.base import (
    EmbeddingProvider,
    LanguageIdProvider,
    NERProvider,
    RerankerProvider,
    TaxonomyMappingProvider,
    TopicProvider,
    VectorIndex,
)
from smap.providers.cache import EmbeddingCacheStore
from smap.providers.embeddings_phobert import PhoBERTEmbeddingProvider
from smap.providers.errors import ProviderUnavailableError
from smap.providers.lid_fasttext import FastTextLanguageIdProvider
from smap.providers.ner_phobert import PhoBERTNERProvider
from smap.providers.semantic_assist import (
    EmbeddingRerankerProvider,
    EmbeddingTaxonomyMappingProvider,
)
from smap.providers.topic_ontology import OntologyGuidedTopicProvider
from smap.providers.vector_faiss import FaissVectorIndex


@dataclass(frozen=True, slots=True)
class ProviderRuntime:
    language_id_provider: LanguageIdProvider
    embedding_provider: EmbeddingProvider
    vector_index: VectorIndex
    topic_provider: TopicProvider
    taxonomy_mapping_provider: TaxonomyMappingProvider
    reranker_provider: RerankerProvider


@dataclass(frozen=True, slots=True)
class SemanticProviderRuntime:
    embedding_provider: EmbeddingProvider
    taxonomy_mapping_provider: TaxonomyMappingProvider
    reranker_provider: RerankerProvider


def language_id_model_candidates(settings: Settings) -> list[Path]:
    configured = settings.intelligence.language_id.fasttext_model_path
    candidates = [
        configured,
        settings.model_dir / configured.name,
        Path("./var/models") / configured.name,
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
    return unique


def resolve_language_id_model_path(settings: Settings) -> Path | None:
    for candidate in language_id_model_candidates(settings):
        if candidate.exists():
            return candidate
    return None


def _module_available(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except ValueError:
        return module_name in sys.modules


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    return _build_embedding_provider(settings, ontology=None)


def _require_embedding_runtime_modules() -> None:
    required_modules = {
        "transformers": "transformers",
        "onnxruntime": "onnxruntime",
        "optimum[onnxruntime]": "optimum.onnxruntime",
    }
    missing = [
        package_name
        for package_name, module_name in required_modules.items()
        if not _module_available(module_name)
    ]
    if missing:
        quoted = ", ".join(f"`{package_name}`" for package_name in missing)
        raise ProviderUnavailableError(f"PhoBERT ONNX embeddings require installed {quoted}.")


def _build_embedding_provider(
    settings: Settings,
    *,
    ontology: OntologyRegistry | None,
) -> EmbeddingProvider:
    if settings.intelligence.embeddings.provider_kind != "phobert":
        raise ProviderUnavailableError("Only PhoBERT embeddings are supported.")
    if settings.intelligence.embeddings.runtime_backend != "onnx":
        raise ProviderUnavailableError("Only the ONNX embedding backend is supported.")
    if settings.intelligence.embeddings.device != "cpu":
        raise ProviderUnavailableError("Only CPU embedding execution is supported.")
    _require_embedding_runtime_modules()
    cache_store = EmbeddingCacheStore(settings.embedding_cache_dir)
    phrase_lexicon = collect_phrase_lexicon(ontology) if ontology is not None else ()
    return PhoBERTEmbeddingProvider(
        model_id=settings.intelligence.embeddings.model_id,
        device=settings.intelligence.embeddings.device,
        batch_size=settings.intelligence.embeddings.batch_size,
        max_length=settings.intelligence.embeddings.max_length,
        cache_store=cache_store,
        phrase_lexicon=phrase_lexicon,
        runtime_backend=settings.intelligence.embeddings.runtime_backend,
        onnx_dir=settings.onnx_model_dir,
        onnx_intra_op_threads=settings.intelligence.embeddings.onnx_intra_op_threads,
        onnx_inter_op_threads=settings.intelligence.embeddings.onnx_inter_op_threads,
    )


def build_language_id_provider(settings: Settings) -> LanguageIdProvider:
    if settings.intelligence.language_id.provider_kind != "fasttext":
        raise ProviderUnavailableError("Only fastText language identification is supported.")
    if not _module_available("fasttext"):
        raise ProviderUnavailableError(
            "fastText language identification is required but the `fasttext` package is unavailable."
        )
    resolved_model_path = resolve_language_id_model_path(settings)
    if resolved_model_path is None:
        raise ProviderUnavailableError(
            "fastText language identification is required but no model file was found."
        )
    lid_settings = settings.intelligence.language_id
    return FastTextLanguageIdProvider(
        model_path=resolved_model_path,
        mixed_confidence_threshold=lid_settings.mixed_confidence_threshold,
        mixed_gap_threshold=lid_settings.mixed_gap_threshold,
        script_override_enabled=lid_settings.script_override_enabled,
    )


def build_vector_index(settings: Settings) -> VectorIndex:
    if settings.intelligence.vector_index.provider_kind != "faiss":
        raise ProviderUnavailableError("Only the FAISS vector index backend is supported.")
    return FaissVectorIndex(storage_dir=settings.vector_index_dir)


def build_topic_provider(
    settings: Settings,
    *,
    ontology: OntologyRegistry | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    preferred_device: str | None = None,
) -> TopicProvider:
    del preferred_device
    if not settings.intelligence.topics.enabled:
        raise ProviderUnavailableError("Ontology-guided topics are required and cannot be disabled.")
    if embedding_provider is None:
        raise ProviderUnavailableError("Ontology-guided topics require the shared PhoBERT embedding provider.")
    registry = ontology or load_runtime_ontology(settings).registry
    if not registry.topics:
        raise ProviderUnavailableError("Selected domain ontology does not define any topics.")
    return OntologyGuidedTopicProvider(ontology=registry, embedding_provider=embedding_provider)


def build_taxonomy_mapping_provider(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> TaxonomyMappingProvider:
    del settings
    if embedding_provider is None:
        raise ProviderUnavailableError("Semantic taxonomy mapping requires the shared PhoBERT embedding provider.")
    return EmbeddingTaxonomyMappingProvider(embedding_provider)


def build_reranker_provider(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> RerankerProvider:
    del settings
    if embedding_provider is None:
        raise ProviderUnavailableError("Semantic reranking requires the shared PhoBERT embedding provider.")
    return EmbeddingRerankerProvider(embedding_provider)


def build_ner_providers(settings: Settings, alias_registry: AliasRegistry) -> list[NERProvider]:
    return build_ner_providers_with_embedding(
        settings,
        alias_registry,
        embedding_provider=build_embedding_provider(settings),
    )


def build_ner_providers_with_embedding(
    settings: Settings,
    alias_registry: AliasRegistry,
    *,
    embedding_provider: EmbeddingProvider,
) -> list[NERProvider]:
    if settings.intelligence.ner.provider_kind != "phobert_ner":
        raise ProviderUnavailableError("Only PhoBERT NER is supported.")
    if embedding_provider.provenance.provider_name != "phobert_embedding":
        raise ProviderUnavailableError("PhoBERT NER requires the shared PhoBERT embedding provider.")
    return [
        PhoBERTNERProvider(
            alias_registry=alias_registry,
            embedding_provider=embedding_provider,
            model_id=settings.intelligence.ner.model_id,
            min_similarity=settings.intelligence.ner.min_similarity,
        )
    ]


def build_provider_runtime(settings: Settings, *, ontology: OntologyRegistry | None = None) -> ProviderRuntime:
    embedding_provider = _build_embedding_provider(settings, ontology=ontology)
    return ProviderRuntime(
        language_id_provider=build_language_id_provider(settings),
        embedding_provider=embedding_provider,
        vector_index=build_vector_index(settings),
        topic_provider=build_topic_provider(
            settings,
            ontology=ontology,
            embedding_provider=embedding_provider,
            preferred_device=embedding_provider.provenance.device,
        ),
        taxonomy_mapping_provider=build_taxonomy_mapping_provider(settings, embedding_provider=embedding_provider),
        reranker_provider=build_reranker_provider(settings, embedding_provider=embedding_provider),
    )


def build_semantic_provider_runtime(
    settings: Settings,
    *,
    ontology: OntologyRegistry | None = None,
) -> SemanticProviderRuntime:
    embedding_provider = _build_embedding_provider(settings, ontology=ontology)
    return SemanticProviderRuntime(
        embedding_provider=embedding_provider,
        taxonomy_mapping_provider=build_taxonomy_mapping_provider(settings, embedding_provider=embedding_provider),
        reranker_provider=build_reranker_provider(settings, embedding_provider=embedding_provider),
    )
