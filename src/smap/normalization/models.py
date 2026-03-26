from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field
from smap.contracts.uap import Platform, UAPType

class MentionRecord(BaseModel):
    mention_id: str
    source_uap_id: str
    origin_id: str
    uap_type: UAPType
    platform: Platform
    project_id: str
    task_id: str
    root_id: str
    parent_id: str | None = None
    depth: int
    author_id: str
    author_username: str | None = None
    author_nickname: str | None = None
    raw_text: str
    normalized_text: str
    normalized_text_compact: str
    language: str
    language_confidence: float = 0.0
    language_provider: str | None = None
    language_provider_version: str | None = None
    language_model_id: str | None = None
    language_source: str = 'inferred'
    language_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    language_supported: bool = True
    language_rejection_reason: str | None = None
    text_quality_label: str = 'normal'
    text_quality_flags: list[str] = Field(default_factory=list)
    text_quality_score: float = 1.0
    mixed_language_uncertain: bool = False
    semantic_route_hint: str = 'semantic_full'
    hashtags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    emojis: list[str] = Field(default_factory=list)
    dedup_cluster_id: str | None = None
    dedup_kind: str | None = None
    dedup_representative_mention_id: str | None = None
    dedup_cluster_size: int = 1
    dedup_similarity: float | None = None
    dedup_weight: float = 1.0
    mention_spam_score: float = 0.0
    author_inorganic_score: float = 0.0
    mention_suspicious: bool = False
    author_suspicious: bool = False
    suspicion_reason_codes: list[str] = Field(default_factory=list)
    quality_weight: float = 1.0
    summary_title: str | None = None
    source_url: str | None = None
    posted_at: datetime | None = None
    ingested_at: datetime | None = None
    likes: int | None = None
    comments_count: int | None = None
    reply_count: int | None = None
    shares: int | None = None
    views: int | None = None
    bookmarks: int | None = None
    sort_score: float | None = None
    is_shop_video: bool | None = None

class NormalizationBatch(BaseModel):
    mentions: list[MentionRecord]
    filtered_out_records: int = 0
    filtered_out_unsupported_language: int = 0
