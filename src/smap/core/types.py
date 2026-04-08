from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

JSONValue = str | int | float | bool | None | dict[str, "JSONValue"] | list["JSONValue"]
RecordDict = dict[str, JSONValue]
Scalar = str | int | float | bool | datetime | UUID | None


def utc_now() -> datetime:
    return datetime.now(UTC)


def coalesce_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

