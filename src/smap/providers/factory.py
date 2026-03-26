from __future__ import annotations
import importlib.util
import inspect
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from smap.canonicalization.alias import AliasRegistry
from smap.core.settings import Settings
from smap.ontology.models import OntologyRegistry
from smap.ontology.prototypes import collect_phrase_lexicon
from smap.ontology.runtime import load_runtime_ontology
from smap.providers.base import EmbeddingProvider, LanguageIdProvider, NERProvider, RerankerProvider, TaxonomyMappingProvider, TopicProvider, VectorIndex
from smap.providers.cache import EmbeddingCacheStore
from smap.providers.embeddings_phobert import PhoBERTEmbeddingProvider
from smap.providers.errors import ProviderUnavailableError
from smap.providers.fallback import FallbackTopicProvider, TokenOverlapEmbeddingProvider
from smap.providers.lid_fasttext import FastTextLanguageIdProvider
from smap.providers.lid_heuristic import HeuristicLanguageIdProvider
from smap.providers.ner_phobert import PhoBERTNERProvider
from smap.providers.semantic_assist import EmbeddingRerankerProvider, EmbeddingTaxonomyMappingProvider
from smap.providers.topic_ontology import OntologyGuidedTopicProvider
from smap.providers.vector_faiss import FaissVectorIndex
from smap.providers.vector_memory import InMemoryVectorIndex

@dataclass(frozen=True, slots=True)
class ProviderRuntime:
    language_id_provider: LanguageIdProvider
    embedding_provider: EmbeddingProvider
    vector_index: VectorIndex
    topic_provider: TopicProvider
    taxonomy_mapping_provider: TaxonomyMappingProvider
    reranker_provider: RerankerProvider

def language_id_model_candidates(settings):
    configured = settings.intelligence.language_id.fasttext_model_path
    candidates = [configured, settings.model_dir / configured.name, Path('./var/models') / configured.name]
    unique = []
    seen = set()
    for candidate in candidates:
        normalized = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
    return unique

def resolve_language_id_model_path(settings):
    for candidate in language_id_model_candidates(settings):
        if candidate.exists():
            return candidate
    return None

def _module_available(module_name):
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except ValueError:
        return module_name in sys.modules

def _preferred_device(requested_device):
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

def build_embedding_provider(settings):
    return _build_embedding_provider(settings, ontology=None)

def _build_embedding_provider(settings, *, ontology):
    cache_store = EmbeddingCacheStore(settings.embedding_cache_dir)
    provider_kind = settings.intelligence.embeddings.provider_kind
    phrase_lexicon = collect_phrase_lexicon(ontology) if ontology is not None else ()
    if settings.intelligence.enable_optional_ml_providers and provider_kind == 'phobert' and _module_available('transformers'):
        model_id = settings.intelligence.embeddings.model_id
        try:
            if 'phrase_lexicon' in inspect.signature(PhoBERTEmbeddingProvider).parameters:
                return PhoBERTEmbeddingProvider(model_id=model_id, device=settings.intelligence.embeddings.device, batch_size=settings.intelligence.embeddings.batch_size, max_length=settings.intelligence.embeddings.max_length, cache_store=cache_store, phrase_lexicon=phrase_lexicon)
            return PhoBERTEmbeddingProvider(model_id=model_id, device=settings.intelligence.embeddings.device, batch_size=settings.intelligence.embeddings.batch_size, max_length=settings.intelligence.embeddings.max_length, cache_store=cache_store)
        except ProviderUnavailableError:
            return TokenOverlapEmbeddingProvider(cache_store=cache_store)
    return TokenOverlapEmbeddingProvider(cache_store=cache_store)

def build_language_id_provider(settings):
    lid_settings = settings.intelligence.language_id
    provider_kind = lid_settings.provider_kind
    resolved_model_path = resolve_language_id_model_path(settings)
    if settings.intelligence.enable_optional_ml_providers and provider_kind == 'fasttext' and _module_available('fasttext') and (resolved_model_path is not None):
        with suppress(ProviderUnavailableError):
            return FastTextLanguageIdProvider(model_path=resolved_model_path, mixed_confidence_threshold=lid_settings.mixed_confidence_threshold, mixed_gap_threshold=lid_settings.mixed_gap_threshold, script_override_enabled=lid_settings.script_override_enabled)
    return HeuristicLanguageIdProvider(mixed_confidence_threshold=lid_settings.mixed_confidence_threshold, mixed_gap_threshold=lid_settings.mixed_gap_threshold, script_override_enabled=lid_settings.script_override_enabled)

def build_vector_index(settings):
    backend = settings.intelligence.vector_index.provider_kind
    if backend == 'faiss':
        try:
            return FaissVectorIndex(storage_dir=settings.vector_index_dir)
        except ProviderUnavailableError:
            return InMemoryVectorIndex(storage_dir=settings.vector_index_dir)
    return InMemoryVectorIndex(storage_dir=settings.vector_index_dir)

def build_topic_provider(settings, *, ontology=None, embedding_provider=None, preferred_device=None):
    del preferred_device
    registry = ontology or load_runtime_ontology(settings).registry
    if not settings.intelligence.topics.enabled or not registry.topics:
        return FallbackTopicProvider()
    return OntologyGuidedTopicProvider(ontology=registry, embedding_provider=embedding_provider)

def build_taxonomy_mapping_provider(settings, *, embedding_provider=None):
    del settings
    return EmbeddingTaxonomyMappingProvider(embedding_provider or TokenOverlapEmbeddingProvider())

def build_reranker_provider(settings, *, embedding_provider=None):
    del settings
    return EmbeddingRerankerProvider(embedding_provider or TokenOverlapEmbeddingProvider())

def build_ner_providers(settings, alias_registry):
    return build_ner_providers_with_embedding(settings, alias_registry, embedding_provider=build_embedding_provider(settings))

def build_ner_providers_with_embedding(settings, alias_registry, *, embedding_provider):
    if settings.intelligence.enable_optional_ml_providers and settings.intelligence.ner.provider_kind == 'phobert_ner' and (embedding_provider.provenance.provider_name == 'phobert_embedding'):
        try:
            return [PhoBERTNERProvider(alias_registry=alias_registry, embedding_provider=embedding_provider, model_id=settings.intelligence.ner.model_id, min_similarity=settings.intelligence.ner.min_similarity)]
        except ProviderUnavailableError:
            return []
    return []

def build_provider_runtime(settings, *, ontology=None):
    embedding_provider = _build_embedding_provider(settings, ontology=ontology)
    embedding_device = embedding_provider.provenance.device or _preferred_device(settings.intelligence.embeddings.device)
    return ProviderRuntime(language_id_provider=build_language_id_provider(settings), embedding_provider=embedding_provider, vector_index=build_vector_index(settings), topic_provider=build_topic_provider(settings, ontology=ontology, embedding_provider=embedding_provider, preferred_device=embedding_device), taxonomy_mapping_provider=build_taxonomy_mapping_provider(settings, embedding_provider=embedding_provider), reranker_provider=build_reranker_provider(settings, embedding_provider=embedding_provider))
