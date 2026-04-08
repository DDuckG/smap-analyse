from __future__ import annotations

from pydantic import BaseModel, Field


class InsightCard(BaseModel):
    insight_type: str
    title: str
    summary: str
    supporting_metrics: dict[str, object]
    evidence_references: list[str] = Field(default_factory=list)
    confidence: float
    time_window: str
    filters_used: dict[str, object] = Field(default_factory=dict)
    source_reports: list[str] = Field(default_factory=list)
