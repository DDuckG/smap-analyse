from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable
from smap.core.types import utc_now
ProviderMetadata = str | int | float | bool | None | list[str]

class EmbeddingPurpose(StrEnum):
    QUERY = 'query'
    PASSAGE = 'passage'
    CLUSTERING = 'clustering'
    LINKING = 'linking'

class LanguageSource(StrEnum):
    EXPLICIT = 'explicit'
    INFERRED = 'inferred'

@dataclass(frozen=True, slots=True)
class ProviderProvenance:
    provider_kind: str
    provider_name: str
    provider_version: str
    model_id: str
    model_revision: str | None = None
    device: str | None = None
    generated_at: datetime = field(default_factory=utc_now)
    run_metadata: dict[str, ProviderMetadata] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class SimilarityMatch:
    candidate_id: str
    score: float
    candidate_text: str | None = None
    metadata: dict[str, ProviderMetadata] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class VectorItem:
    item_id: str
    vector: tuple[float, ...]
    text: str
    metadata: dict[str, ProviderMetadata] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    item_id: str
    score: float
    text: str
    metadata: dict[str, ProviderMetadata] = field(default_factory=dict)

class VectorReuseState(StrEnum):
    VALID = 'valid'
    STALE = 'stale'
    INCOMPATIBLE = 'incompatible'
    MISSING = 'missing'
    REFRESH_REQUIRED = 'refresh_required'

@dataclass(frozen=True, slots=True)
class VectorNamespaceExpectation:
    namespace: str
    backend: str | None = None
    dimension: int | None = None
    normalization_mode: str | None = None
    embedding_model_id: str | None = None
    embedding_provider_name: str | None = None
    embedding_provider_version: str | None = None
    embedding_purpose: str | None = None
    corpus_hash: str | None = None

@dataclass(frozen=True, slots=True)
class VectorNamespaceInfo:
    namespace: str
    backend: str
    item_count: int
    dimension: int
    normalization_mode: str
    embedding_model_id: str | None = None
    embedding_provider_name: str | None = None
    embedding_provider_version: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    corpus_hash: str | None = None
    expected_corpus_hash: str | None = None
    storage_path: str | None = None
    reuse_state: str = VectorReuseState.VALID.value
    compatibility_errors: list[str] = field(default_factory=list)
    recommended_action: str | None = None
    metadata: dict[str, ProviderMetadata] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class RecognizedEntitySpan:
    start: int
    end: int
    text: str
    normalized_text: str
    label: str
    entity_type_hint: str | None
    confidence: float
    provider_provenance: ProviderProvenance
    metadata: dict[str, ProviderMetadata] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TopicDocument:
    document_id: str
    mention_id: str
    source_uap_id: str
    text: str
    normalized_text: str
    segment_id: str | None = None
    metadata: dict[str, ProviderMetadata] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TopicAssignment:
    document_id: str
    topic_key: str
    topic_label: str
    confidence: float
    representative: bool = False
    metadata: dict[str, ProviderMetadata] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TopicArtifact:
    topic_key: str
    topic_label: str
    top_terms: tuple[str, ...]
    representative_document_ids: tuple[str, ...]
    topic_size: int
    provider_provenance: ProviderProvenance
    time_window_start: str | None = None
    time_window_end: str | None = None
    growth_delta: float | None = None
    metadata: dict[str, ProviderMetadata] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TopicDiscoveryResult:
    assignments: list[TopicAssignment]
    artifacts: list[TopicArtifact]

@dataclass(frozen=True, slots=True)
class TaxonomyMappingCandidate:
    source_text: str
    taxonomy_label: str
    score: float
    provider_provenance: ProviderProvenance
    metadata: dict[str, ProviderMetadata] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class LanguageIdentificationResult:
    language: str
    confidence: float
    provider_provenance: ProviderProvenance
    source: LanguageSource = LanguageSource.INFERRED
    metadata: dict[str, ProviderMetadata] = field(default_factory=dict)

@runtime_checkable
class EmbeddingProvider(Protocol):
    version: str
    provenance: ProviderProvenance

    def embed_texts(self, texts, *, purpose=EmbeddingPurpose.PASSAGE):
        ...

    def rank_candidates(self, text, candidates, *, purpose=EmbeddingPurpose.LINKING, top_k=5):
        ...

    def best_match(self, text, candidates):
        ...
EmbeddingProviderV2 = EmbeddingProvider

@runtime_checkable
class VectorIndex(Protocol):
    version: str
    provenance: ProviderProvenance

    def reset(self, *, namespace):
        ...

    def bind_expectation(self, *, namespace, expected):
        ...

    def expectation_for(self, *, namespace):
        ...

    def load(self, *, namespace, expected=None, allow_stale=False, force=False):
        ...

    def upsert(self, items, *, namespace):
        ...

    def info(self, *, namespace, expected=None):
        ...

    def search(self, vector, *, namespace, top_k=5):
        ...

@runtime_checkable
class NERProvider(Protocol):
    version: str
    provenance: ProviderProvenance

    def extract(self, text, *, mention_id, source_uap_id, label_inventory=None):
        ...

@runtime_checkable
class TopicProvider(Protocol):
    version: str
    provenance: ProviderProvenance

    def discover(self, documents, *, embeddings=None):
        ...

@runtime_checkable
class TaxonomyMappingProvider(Protocol):
    version: str
    provenance: ProviderProvenance

    def map_labels(self, labels, taxonomy_labels):
        ...

    def map_candidates(self, labels, taxonomy_labels, *, top_k=3):
        ...

@runtime_checkable
class RerankerProvider(Protocol):
    version: str
    provenance: ProviderProvenance

    def rerank(self, query, candidates):
        ...

class ZeroShotClassifierProvider(Protocol):
    version: str

    def classify(self, text, labels):
        ...

@runtime_checkable
class LanguageIdProvider(Protocol):
    version: str
    provenance: ProviderProvenance

    def detect(self, text):
        ...
