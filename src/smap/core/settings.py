from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class EmbeddingSettings(BaseModel):
    provider_kind: str = 'phobert'
    model_id: str = 'vinai/phobert-base-v2'
    device: str = 'auto'
    batch_size: int = 16
    max_length: int = 256
    cache_dirname: str = 'embedding_cache'

class LanguageIdSettings(BaseModel):
    provider_kind: str = 'fasttext'
    fallback_provider_kind: str = 'heuristic'
    fasttext_model_path: Path = Field(default=Path('./var/data/models/lid.176.ftz'))
    mixed_confidence_threshold: float = 0.55
    mixed_gap_threshold: float = 0.12
    script_override_enabled: bool = True

class VectorIndexSettings(BaseModel):
    provider_kind: str = 'faiss'
    fallback_provider_kind: str = 'memory'
    namespace: str = 'default'
    storage_dirname: str = 'vector_index'

class NERSettings(BaseModel):
    provider_kind: str = 'phobert_ner'
    model_id: str = 'vinai/phobert-base-v2'
    min_similarity: float = 0.58
    label_inventory: list[str] = Field(default_factory=lambda: ['brand', 'product', 'person', 'organization', 'location', 'facility', 'retailer', 'concept'])

class TopicSettings(BaseModel):
    provider_kind: str = 'ontology_guided'
    enabled: bool = True
    secondary_discovery_enabled: bool = False
    artifact_dirname: str = 'topics'

class IntelligenceSettings(BaseModel):
    enable_optional_ml_providers: bool = True
    semantic_assist_enabled: bool = True
    semantic_hypothesis_rerank_enabled: bool = True
    semantic_corroboration_enabled: bool = True
    entity_embedding_rerank_enabled: bool = True
    topic_label_rerank_enabled: bool = True
    language_id: LanguageIdSettings = Field(default_factory=LanguageIdSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_index: VectorIndexSettings = Field(default_factory=VectorIndexSettings)
    ner: NERSettings = Field(default_factory=NERSettings)
    topics: TopicSettings = Field(default_factory=TopicSettings)

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
    weighting_mode: str = 'raw'
    dedup: DedupSettings = Field(default_factory=DedupSettings)
    spam: SpamSettings = Field(default_factory=SpamSettings)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_prefix='SMAP_', case_sensitive=False, extra='ignore')
    env: str = 'local'
    data_dir: Path = Field(default=Path('./var/data'))
    db_url: str = 'sqlite:///./var/app.db'
    analytics_duckdb: Path = Field(default=Path('./var/analytics.duckdb'))
    domain_ontology_dir: Path = Field(default=Path('./configs/domains'))
    domain_ontology_path: Path | None = None
    intelligence: IntelligenceSettings = Field(default_factory=IntelligenceSettings)
    analytics: AnalyticsSettings = Field(default_factory=AnalyticsSettings)

    @property
    def bronze_dir(self):
        return self.data_dir / 'bronze'

    @property
    def silver_dir(self):
        return self.data_dir / 'silver'

    @property
    def gold_dir(self):
        return self.data_dir / 'gold'

    @property
    def reports_dir(self):
        return self.data_dir / 'reports'

    @property
    def insights_dir(self):
        return self.data_dir / 'insights'

    @property
    def intelligence_dir(self):
        return self.data_dir / 'intelligence'

    @property
    def model_dir(self):
        return self.data_dir / 'models'

    @property
    def embedding_cache_dir(self):
        return self.intelligence_dir / self.intelligence.embeddings.cache_dirname

    @property
    def vector_index_dir(self):
        return self.intelligence_dir / self.intelligence.vector_index.storage_dirname

    @property
    def topic_artifacts_dir(self):
        return self.intelligence_dir / self.intelligence.topics.artifact_dirname

    @property
    def feedback_dir(self):
        return self.intelligence_dir / 'feedback'

    @property
    def semantic_promoted_path(self):
        return self.feedback_dir / 'promoted_semantic_knowledge.jsonl'

    @property
    def topic_lineage_path(self):
        return self.feedback_dir / 'promoted_topic_lineage.jsonl'

    @property
    def latest_topic_artifact_snapshot_path(self):
        return self.topic_artifacts_dir / 'latest_topic_artifacts.jsonl'

    @property
    def sqlite_db_path(self):
        prefix = 'sqlite:///'
        if self.db_url.startswith(prefix):
            return Path(self.db_url.replace(prefix, ''))
        return None

    def available_domain_ontology_paths(self):
        if not self.domain_ontology_dir.exists():
            return []
        return sorted((path.resolve() for path in self.domain_ontology_dir.glob('*.yaml')))

    def ensure_directories(self):
        db_path = self.sqlite_db_path
        for path in (self.data_dir, self.bronze_dir, self.silver_dir, self.gold_dir, self.reports_dir, self.insights_dir, self.intelligence_dir, self.model_dir, self.embedding_cache_dir, self.vector_index_dir, self.topic_artifacts_dir, self.feedback_dir, self.analytics_duckdb, self.intelligence.language_id.fasttext_model_path, db_path if db_path is not None else Path('.')):
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                continue
            path.mkdir(parents=True, exist_ok=True)

def get_settings():
    return Settings()
