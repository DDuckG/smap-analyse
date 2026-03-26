from __future__ import annotations
from pydantic import BaseModel, Field

class DedupClusterRecord(BaseModel):
    dedup_cluster_id: str
    dedup_kind: str
    representative_mention_id: str
    representative_text: str
    mention_ids: list[str] = Field(default_factory=list)
    cluster_size: int
    similarity_proxy: float
    text_fingerprint: str | None = None

class DedupAnalysisResult(BaseModel):
    mentions_updated: int = 0
    exact_cluster_count: int = 0
    near_cluster_count: int = 0
    clusters: list[DedupClusterRecord] = Field(default_factory=list)
