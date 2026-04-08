from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from smap.ontology.domain_mapping import legacy_overlay_domain_path


class EmbeddingSettings(BaseModel):
    provider_kind: Literal["phobert"] = "phobert"
    model_id: str = "vinai/phobert-base-v2"
    device: Literal["cpu"] = "cpu"
    batch_size: int = Field(default=24, ge=1, le=128)
    max_length: int = Field(default=256, ge=32, le=1024)
    cache_dirname: str = "embedding_cache"
    runtime_backend: Literal["onnx"] = "onnx"
    onnx_dirname: str = "onnx_models"
    onnx_dir_override: Path | None = None
    onnx_intra_op_threads: int | None = Field(default=None, ge=1)
    onnx_inter_op_threads: int | None = Field(default=None, ge=1)


class LanguageIdSettings(BaseModel):
    provider_kind: Literal["fasttext"] = "fasttext"
    fasttext_model_path: Path = Field(default=Path("./var/data/models/lid.176.ftz"))
    min_text_length: int = 3
    mixed_confidence_threshold: float = 0.55
    mixed_gap_threshold: float = 0.12
    script_override_enabled: bool = True


class VectorIndexSettings(BaseModel):
    provider_kind: Literal["faiss"] = "faiss"
    namespace: str = "default"
    storage_dirname: str = "vector_index"


class NERSettings(BaseModel):
    provider_kind: Literal["phobert_ner"] = "phobert_ner"
    model_id: str = "vinai/phobert-base-v2"
    min_similarity: float = 0.58
    label_inventory: list[str] = Field(
        default_factory=lambda: [
            "brand",
            "product",
            "person",
            "organization",
            "location",
            "facility",
            "retailer",
            "concept",
        ]
    )


class TopicSettings(BaseModel):
    provider_kind: Literal["ontology_guided"] = "ontology_guided"
    enabled: bool = True
    secondary_discovery_enabled: bool = False
    artifact_dirname: str = "topics"


class LabelStudioSettings(BaseModel):
    export_dirname: str = "label_studio/exports"
    import_dirname: str = "label_studio/imports"


VerifierLevel = Literal["a", "b", "c"]


class VerifierEscalationSettings(BaseModel):
    entity_level: VerifierLevel = "b"
    topic_artifact_level: VerifierLevel = "b"
    lineage_level: VerifierLevel = "b"


class IntelligenceSettings(BaseModel):
    semantic_assist_enabled: bool = True
    semantic_hypothesis_rerank_enabled: bool = True
    semantic_corroboration_enabled: bool = True
    semantic_runtime_diagnostics_enabled: bool = False
    semantic_orchestration_cache_enabled: bool = True
    entity_embedding_rerank_enabled: bool = True
    semantic_parallel_workers: int = Field(default=2, ge=1, le=16)
    semantic_parallel_min_mentions: int = Field(default=64, ge=1)
    semantic_parallel_chunk_size: int = Field(default=32, ge=1)
    language_id: LanguageIdSettings = Field(default_factory=LanguageIdSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_index: VectorIndexSettings = Field(default_factory=VectorIndexSettings)
    ner: NERSettings = Field(default_factory=NERSettings)
    topics: TopicSettings = Field(default_factory=TopicSettings)
    label_studio: LabelStudioSettings = Field(default_factory=LabelStudioSettings)
    verifiers: VerifierEscalationSettings = Field(default_factory=VerifierEscalationSettings)


class DedupSettings(BaseModel):
    enabled: bool = True
    exact_enabled: bool = True
    near_enabled: bool = True
    min_text_length: int = 10
    word_shingle_size: int = 3
    char_shingle_size: int = 5
    num_perm: int = 64
    num_bands: int = 16
    near_similarity_threshold: float = 0.82
    max_bucket_size: int = 256


class SpamSettings(BaseModel):
    enabled: bool = True
    mention_threshold: float = 0.65
    author_threshold: float = 0.65
    burst_window_minutes: int = 60
    quality_weight_floor: float = 0.15
    mention_discount_strength: float = 0.65
    author_discount_strength: float = 0.45


class AnalyticsSettings(BaseModel):
    weighting_mode: str = "raw"
    dedup: DedupSettings = Field(default_factory=DedupSettings)
    spam: SpamSettings = Field(default_factory=SpamSettings)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SMAP_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    env: str = "local"
    data_dir: Path = Field(default=Path("./var/data"))
    db_url: str = "sqlite:///./var/app.db"
    analytics_duckdb: Path = Field(default=Path("./var/analytics.duckdb"))
    domain_ontology_dir: Path = Field(default=Path("./configs/domains"))
    domain_ontology_path: Path | None = None
    domain_id: str | None = None
    ontology_path: Path = Field(default=Path("./configs/ontology.yaml"))
    ontology_overlays: str = ""
    review_overlay_path: Path = Field(default=Path("./var/review/reviewed_aliases.yaml"))
    intelligence: IntelligenceSettings = Field(default_factory=IntelligenceSettings)
    analytics: AnalyticsSettings = Field(default_factory=AnalyticsSettings)

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def insights_dir(self) -> Path:
        return self.data_dir / "insights"

    @property
    def review_dir(self) -> Path:
        return self.review_overlay_path.parent

    @property
    def intelligence_dir(self) -> Path:
        return self.data_dir / "intelligence"

    @property
    def embedding_cache_dir(self) -> Path:
        return self.intelligence_dir / self.intelligence.embeddings.cache_dirname

    @property
    def onnx_model_dir(self) -> Path:
        override = self.intelligence.embeddings.onnx_dir_override
        if override is not None:
            return override
        return self.intelligence_dir / self.intelligence.embeddings.onnx_dirname

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def vector_index_dir(self) -> Path:
        return self.intelligence_dir / self.intelligence.vector_index.storage_dirname

    @property
    def topic_artifacts_dir(self) -> Path:
        return self.intelligence_dir / self.intelligence.topics.artifact_dirname

    @property
    def feedback_dir(self) -> Path:
        return self.intelligence_dir / "feedback"

    @property
    def semantic_feedback_path(self) -> Path:
        return self.feedback_dir / "approved_semantic_annotations.jsonl"

    @property
    def semantic_promoted_path(self) -> Path:
        return self.feedback_dir / "promoted_semantic_knowledge.jsonl"

    @property
    def semantic_benchmark_gold_path(self) -> Path:
        return self.feedback_dir / "semantic_benchmark_gold.jsonl"

    @property
    def topic_feedback_path(self) -> Path:
        return self.feedback_dir / "approved_topic_reviews.jsonl"

    @property
    def topic_lineage_path(self) -> Path:
        return self.feedback_dir / "promoted_topic_lineage.jsonl"

    @property
    def latest_topic_artifact_snapshot_path(self) -> Path:
        return self.topic_artifacts_dir / "latest_topic_artifacts.jsonl"

    @property
    def label_studio_export_dir(self) -> Path:
        return self.data_dir / self.intelligence.label_studio.export_dirname

    @property
    def label_studio_import_dir(self) -> Path:
        return self.data_dir / self.intelligence.label_studio.import_dirname

    @property
    def sqlite_db_path(self) -> Path | None:
        prefix = "sqlite:///"
        if self.db_url.startswith(prefix):
            return Path(self.db_url.replace(prefix, ""))
        return None

    @property
    def ontology_overlay_paths(self) -> list[Path]:
        if not self.ontology_overlays.strip():
            return []
        return [
            Path(item.strip())
            for item in re.split(r"[;,]", self.ontology_overlays)
            if item.strip()
        ]

    @property
    def all_ontology_overlay_paths(self) -> list[Path]:
        overlay_paths = list(self.ontology_overlay_paths)
        if self.review_overlay_path.exists():
            overlay_paths.append(self.review_overlay_path)
        return overlay_paths

    @property
    def legacy_domain_ontology_path(self) -> Path | None:
        return legacy_overlay_domain_path(self.ontology_overlay_paths, self.domain_ontology_dir)

    @property
    def selected_domain_ontology_path(self) -> Path:
        if self.domain_ontology_path is not None:
            return self.domain_ontology_path
        if self.domain_id:
            return self.domain_ontology_dir / f"{self.domain_id}.yaml"
        legacy_domain = self.legacy_domain_ontology_path
        if legacy_domain is not None:
            return legacy_domain
        return self.domain_ontology_dir / "cosmetics_vn.yaml"

    def available_domain_ontology_paths(self) -> list[Path]:
        if not self.domain_ontology_dir.exists():
            return []
        return sorted(path.resolve() for path in self.domain_ontology_dir.glob("*.yaml"))

    def ensure_directories(self) -> None:
        db_path = self.sqlite_db_path
        for path in (
            self.data_dir,
            self.bronze_dir,
            self.silver_dir,
            self.gold_dir,
            self.reports_dir,
            self.insights_dir,
            self.review_dir,
            self.intelligence_dir,
            self.model_dir,
            self.embedding_cache_dir,
            self.onnx_model_dir,
            self.vector_index_dir,
            self.topic_artifacts_dir,
            self.feedback_dir,
            self.label_studio_export_dir,
            self.label_studio_import_dir,
            Path(self.analytics_duckdb).parent,
            db_path if db_path is not None else Path("."),
            self.review_overlay_path,
            self.intelligence.language_id.fasttext_model_path,
        ):
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
