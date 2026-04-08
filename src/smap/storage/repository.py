from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import duckdb
import polars as pl
from pydantic import BaseModel


def _normalize_model_records(records: Iterable[BaseModel]) -> list[dict[str, object]]:
    return [record.model_dump(mode="json") for record in records]


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    write_text_if_changed(path, payload)


def write_text_if_changed(path: Path, payload: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == payload:
            return False
    path.write_text(payload, encoding="utf-8")
    return True


def copy_file_if_needed(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return False
    shutil.copyfile(source, destination)
    return True


def write_models_parquet(path: Path, records: Iterable[BaseModel]) -> pl.DataFrame:
    materialized = _normalize_model_records(records)
    return write_dicts_parquet(path, materialized)


def write_dicts_parquet(path: Path, records: list[dict[str, object]]) -> pl.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = (
        pl.DataFrame(records, infer_schema_length=len(records))
        if records
        else pl.DataFrame([])
    )
    frame.write_parquet(path)
    return frame


def write_table_bundle(output_dir: Path, tables: dict[str, pl.DataFrame]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths: dict[str, Path] = {}
    for table_name, table in tables.items():
        path = output_dir / f"{table_name}.parquet"
        table.write_parquet(path)
        table_paths[table_name] = path
    return table_paths


def register_parquet_tables(duckdb_path: Path, table_paths: dict[str, Path]) -> None:
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(duckdb_path))
    try:
        for table_name, table_path in table_paths.items():
            connection.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet(?)",
                [str(table_path)],
            )
    finally:
        connection.close()
