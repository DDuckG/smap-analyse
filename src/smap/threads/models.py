from __future__ import annotations
from pydantic import BaseModel, Field

class ThreadEdge(BaseModel):
    root_id: str
    parent_id: str
    child_id: str
    depth: int

class MentionContext(BaseModel):
    mention_id: str
    root_id: str
    parent_id: str | None = None
    lineage_ids: list[str] = Field(default_factory=list)
    sibling_ids: list[str] = Field(default_factory=list)
    direct_child_ids: list[str] = Field(default_factory=list)
    context_text: str
    root_text: str
    parent_text: str | None = None

class ThreadSummary(BaseModel):
    root_id: str
    total_mentions: int
    total_descendants: int
    max_depth_observed: int
    comment_count: int
    reply_count: int
    top_comment_ids: list[str] = Field(default_factory=list)
    top_comment_scores: list[float] = Field(default_factory=list)

class ThreadBundle(BaseModel):
    summaries: list[ThreadSummary]
    edges: list[ThreadEdge]
    contexts: list[MentionContext]
