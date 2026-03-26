from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from pydantic import ValidationError
from smap.contracts.uap import parse_uap_record
from smap.ingestion.models import IngestedBatchBundle
from smap.ingestion.readers import iter_batch_source
from smap.validation.models import BatchProfile, FieldProfile, ValidationErrorItem, ValidationReport, counter_to_dict, flatten_payload

@dataclass(slots=True)
class FieldStats:
    present: int = 0
    missing: int = 0
    examples: list[str] = field(default_factory=list)

def validate_batch(path):
    if isinstance(path, IngestedBatchBundle):
        return path.validation_report.model_copy(deep=True)
    report = ValidationReport()
    records_by_type = Counter()
    for item in iter_batch_source(path):
        report.total_records += 1
        try:
            record = parse_uap_record(item.payload)
        except (ValidationError, ValueError) as exc:
            report.invalid_records += 1
            report.errors.append(ValidationErrorItem(source_path=item.source_path, line_number=item.line_number, error=str(exc)))
            continue
        report.valid_records += 1
        records_by_type[record.identity.uap_type.value] += 1
    report.records_by_type = counter_to_dict(records_by_type)
    return report

def profile_batch(path):
    if isinstance(path, IngestedBatchBundle):
        return path.batch_profile.model_copy(deep=True)
    records_by_type = Counter()
    records_by_platform = Counter()
    field_stats = defaultdict(FieldStats)
    text_lengths = []
    notes = []
    total_records = 0
    for item in iter_batch_source(path):
        total_records += 1
        payload = item.payload
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
    if total_records and field_stats.get('content.language', FieldStats()).present == 0:
        notes.append('content.language is absent in the sampled batch; fallback detection is required.')
    if total_records and field_stats.get('content.tiktok_keywords', FieldStats()).present < total_records:
        notes.append('TikTok AI-derived fields are sparse and must remain optional.')
    profile = BatchProfile(total_records=total_records, records_by_type=counter_to_dict(records_by_type), records_by_platform=counter_to_dict(records_by_platform), field_profiles=[FieldProfile(field_path=field_name, present_count=stats.present, missing_count=stats.missing, example_values=stats.examples) for field_name, stats in sorted(field_stats.items())], text_length={'avg': round(mean(text_lengths), 2) if text_lengths else 0.0, 'min': float(min(text_lengths)) if text_lengths else 0.0, 'max': float(max(text_lengths)) if text_lengths else 0.0}, notes=notes)
    return profile
