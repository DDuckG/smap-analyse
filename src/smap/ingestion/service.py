from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from pydantic import ValidationError
from smap.contracts.uap import parse_uap_record
from smap.ingestion.models import IngestedBatchBundle
from smap.ingestion.readers import iter_batch_source
from smap.run_manifest import inspect_input_path
from smap.validation.models import BatchProfile, FieldProfile, ValidationErrorItem, ValidationReport, counter_to_dict, flatten_payload

@dataclass(slots=True)
class _FieldStats:
    present: int = 0
    missing: int = 0
    examples: list[str] = field(default_factory=list)

def ingest_batch(input_path, *, materialized_raw_jsonl_path=None):
    validation_report = ValidationReport()
    records_by_type = Counter()
    records_by_platform = Counter()
    field_stats = defaultdict(_FieldStats)
    text_lengths = []
    notes = []
    parsed_records = []
    raw_output_path = materialized_raw_jsonl_path.resolve() if materialized_raw_jsonl_path is not None else None
    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_handle_context = raw_output_path.open('w', encoding='utf-8') if raw_output_path is not None else nullcontext()
    with output_handle_context as raw_handle:
        for item in iter_batch_source(input_path):
            payload = item.payload
            if raw_handle is not None:
                raw_handle.write(item.raw_line)
            validation_report.total_records += 1
            flat = flatten_payload(payload)
            identity = payload.get('identity', {})
            if isinstance(identity, dict):
                if 'uap_type' in identity:
                    records_by_type[str(identity['uap_type'])] += 1
                if 'platform' in identity:
                    records_by_platform[str(identity['platform'])] += 1
            for field_name, value in flat.items():
                stats = field_stats[field_name]
                if value in (None, '', [], {}):
                    stats.missing += 1
                else:
                    stats.present += 1
                    if len(stats.examples) < 3:
                        stats.examples.append(str(value)[:80])
            text_value = flat.get('content.text')
            if isinstance(text_value, str):
                text_lengths.append(len(text_value))
            try:
                record = parse_uap_record(payload)
            except (ValidationError, ValueError) as exc:
                validation_report.invalid_records += 1
                validation_report.errors.append(ValidationErrorItem(source_path=item.source_path, line_number=item.line_number, error=str(exc)))
                continue
            validation_report.valid_records += 1
            parsed_records.append(record)
    if validation_report.total_records and field_stats.get('content.language', _FieldStats()).present == 0:
        notes.append('content.language is absent in the sampled batch; fallback detection is required.')
    if validation_report.total_records and field_stats.get('content.tiktok_keywords', _FieldStats()).present < validation_report.total_records:
        notes.append('TikTok AI-derived fields are sparse and must remain optional.')
    batch_profile = BatchProfile(total_records=validation_report.total_records, records_by_type=counter_to_dict(records_by_type), records_by_platform=counter_to_dict(records_by_platform), field_profiles=[FieldProfile(field_path=field_name, present_count=stats.present, missing_count=stats.missing, example_values=stats.examples) for field_name, stats in sorted(field_stats.items())], text_length={'avg': round(mean(text_lengths), 2) if text_lengths else 0.0, 'min': float(min(text_lengths)) if text_lengths else 0.0, 'max': float(max(text_lengths)) if text_lengths else 0.0}, notes=notes)
    validation_report.records_by_type = counter_to_dict(records_by_type)
    return IngestedBatchBundle(input_path=str(input_path), parsed_records=parsed_records, validation_report=validation_report, batch_profile=batch_profile, input_inspection=inspect_input_path(input_path), materialized_raw_jsonl_path=str(raw_output_path) if raw_output_path is not None else None, raw_record_count=validation_report.total_records)
