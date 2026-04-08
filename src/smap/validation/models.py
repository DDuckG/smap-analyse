from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field


class ValidationErrorItem(BaseModel):
    source_path: str
    line_number: int
    error: str


class ValidationReport(BaseModel):
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    records_by_type: dict[str, int] = Field(default_factory=dict)
    errors: list[ValidationErrorItem] = Field(default_factory=list)


class FieldProfile(BaseModel):
    field_path: str
    present_count: int = 0
    missing_count: int = 0
    example_values: list[str] = Field(default_factory=list)


class BatchProfile(BaseModel):
    total_records: int = 0
    records_by_type: dict[str, int] = Field(default_factory=dict)
    records_by_platform: dict[str, int] = Field(default_factory=dict)
    field_profiles: list[FieldProfile] = Field(default_factory=list)
    text_length: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Batch Profile",
            "",
            f"- Total records: {self.total_records}",
            f"- Types: {self.records_by_type}",
            f"- Platforms: {self.records_by_platform}",
            "",
            "| Field | Present | Missing | Examples |",
            "| --- | ---: | ---: | --- |",
        ]
        for field_profile in self.field_profiles:
            lines.append(
                "| {field} | {present} | {missing} | {examples} |".format(
                    field=field_profile.field_path,
                    present=field_profile.present_count,
                    missing=field_profile.missing_count,
                    examples=", ".join(field_profile.example_values[:3]) or "-",
                )
            )
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


def counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(counter)


def flatten_payload(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_payload(value, prefix=path))
        else:
            flat[path] = value
    return flat

