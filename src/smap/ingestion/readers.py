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

def iter_jsonl_file(path):
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield SourceRecord(source_path=str(path), line_number=line_number, raw_line=line if line.endswith('\n') else f'{line}\n', payload=json.loads(line))

def iter_zip_jsonl(path):
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.jsonl'):
                continue
            with archive.open(name, 'r') as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.decode('utf-8')
                    if not line.strip():
                        continue
                    yield SourceRecord(source_path=f'{path}!{name}', line_number=line_number, raw_line=line if line.endswith('\n') else f'{line}\n', payload=json.loads(line))

def iter_batch_source(path):
    if path.is_file() and path.suffix == '.jsonl':
        yield from iter_jsonl_file(path)
        return
    if path.is_file() and path.suffix == '.zip':
        yield from iter_zip_jsonl(path)
        return
    if path.is_dir():
        for jsonl_path in sorted(path.rglob('*.jsonl')):
            yield from iter_jsonl_file(jsonl_path)
        return
    raise FileNotFoundError(path)

def read_validated_records(path):
    return [parse_uap_record(item.payload) for item in iter_batch_source(path)]
