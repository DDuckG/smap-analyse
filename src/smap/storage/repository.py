from __future__ import annotations
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
import duckdb
import polars as pl
from pydantic import BaseModel

def _normalize_model_records(records):
    return [record.model_dump(mode='json') for record in records]

def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join((json.dumps(record, ensure_ascii=False) + '\n' for record in records))
    write_text_if_changed(path, payload)

def write_text_if_changed(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding='utf-8')
        if existing == payload:
            return False
    path.write_text(payload, encoding='utf-8')
    return True

def copy_file_if_needed(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return False
    shutil.copyfile(source, destination)
    return True

def write_models_parquet(path, records):
    materialized = _normalize_model_records(records)
    return write_dicts_parquet(path, materialized)

def write_dicts_parquet(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(records, infer_schema_length=len(records)) if records else pl.DataFrame([])
    frame.write_parquet(path)
    return frame

def write_table_bundle(output_dir, tables):
    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths = {}
    for table_name, table in tables.items():
        path = output_dir / f'{table_name}.parquet'
        table.write_parquet(path)
        table_paths[table_name] = path
    return table_paths

def register_parquet_tables(duckdb_path, table_paths):
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(duckdb_path))
    try:
        for table_name, table_path in table_paths.items():
            connection.execute(f'CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet(?)', [str(table_path)])
    finally:
        connection.close()
