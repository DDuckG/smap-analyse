from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from smap.contracts.uap import ParsedUAPRecord
from smap.run_manifest import InputInspection
from smap.validation.models import BatchProfile, ValidationReport


class IngestedBatchBundle(BaseModel):
    input_path: str
    parsed_records: list[ParsedUAPRecord] = Field(default_factory=list)
    validation_report: ValidationReport
    batch_profile: BatchProfile
    input_inspection: InputInspection
    materialized_raw_jsonl_path: str | None = None
    raw_record_count: int = 0

    @property
    def path(self) -> Path:
        return Path(self.input_path)

    @property
    def raw_jsonl_path(self) -> Path | None:
        if self.materialized_raw_jsonl_path is None:
            return None
        return Path(self.materialized_raw_jsonl_path)
