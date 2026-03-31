from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from smap.contracts.uap import ParsedUAPRecord, parse_uap_record


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_path: str
    line_number: int
    raw_line: str
    payload: dict[str, object]


def is_valid_jsonl_archive_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if not normalized.endswith(".jsonl"):
        return False
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return False
    if "__MACOSX" in parts:
        return False
    return not any(part.startswith("._") for part in parts)


def iter_jsonl_archive_members(archive: ZipFile) -> Iterator[str]:
    for name in sorted(archive.namelist()):
        if is_valid_jsonl_archive_member(name):
            yield name


def decode_jsonl_archive_line(raw_line: bytes, *, source_path: str, line_number: int) -> str:
    try:
        return raw_line.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Archive member {source_path} contains a non-UTF-8 JSONL line at {line_number}."
        ) from exc


def iter_jsonl_file(path: Path) -> Iterator[SourceRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield SourceRecord(
                source_path=str(path),
                line_number=line_number,
                raw_line=line if line.endswith("\n") else f"{line}\n",
                payload=json.loads(line),
            )


def iter_zip_jsonl(path: Path) -> Iterator[SourceRecord]:
    with ZipFile(path) as archive:
        for name in iter_jsonl_archive_members(archive):
            with archive.open(name, "r") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = decode_jsonl_archive_line(
                        raw_line,
                        source_path=f"{path}!{name}",
                        line_number=line_number,
                    )
                    if not line.strip():
                        continue
                    yield SourceRecord(
                        source_path=f"{path}!{name}",
                        line_number=line_number,
                        raw_line=line if line.endswith("\n") else f"{line}\n",
                        payload=json.loads(line),
                    )


def iter_batch_source(path: Path) -> Iterator[SourceRecord]:
    if path.is_file() and path.suffix == ".jsonl":
        yield from iter_jsonl_file(path)
        return
    if path.is_file() and path.suffix == ".zip":
        yield from iter_zip_jsonl(path)
        return
    if path.is_dir():
        for jsonl_path in sorted(path.rglob("*.jsonl")):
            yield from iter_jsonl_file(jsonl_path)
        return
    raise FileNotFoundError(path)


def read_validated_records(path: Path) -> list[ParsedUAPRecord]:
    return [parse_uap_record(item.payload) for item in iter_batch_source(path)]
