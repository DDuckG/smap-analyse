from __future__ import annotations

from pydantic import BaseModel, Field


class AuthorQualityRecord(BaseModel):
    author_id: str
    mention_count: int
    suspicious_mention_count: int
    author_inorganic_score: float
    author_suspicious: bool
    burstiness_score: float
    template_ratio: float
    duplicate_cluster_participation_rate: float
    repost_like_ratio: float
    average_mention_spam_score: float
    reason_codes: list[str] = Field(default_factory=list)


class QualityAnalysisResult(BaseModel):
    mentions_updated: int = 0
    author_profiles: list[AuthorQualityRecord] = Field(default_factory=list)
