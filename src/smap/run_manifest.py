from pathlib import Path
from typing import Literal
from zipfile import ZipFile
from pydantic import BaseModel, Field
from smap.bi.contracts import BI_SCHEMA_VERSION
from smap.core.types import utc_now
from smap.runtime.doctor import run_runtime_doctor
RUN_MANIFEST_VERSION = '2026.03.26-core-handoff-v1'

class InputBatchEntry(BaseModel):
    source_path: str
    kind: Literal['jsonl']

class InputInspection(BaseModel):
    input_path: str
    input_kind: Literal['jsonl', 'zip', 'directory', 'missing', 'unknown']
    jsonl_entry_count: int = 0
    total_container_entries: int | None = None
    batch_entries: list[InputBatchEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class RuntimeModeSummary(BaseModel):
    requested_mode: Literal['default_intelligence', 'degraded_fallback']
    effective_mode: Literal['default_intelligence', 'degraded_fallback']
    provider_modes: dict[str, str] = Field(default_factory=dict)
    provider_provenance: dict[str, dict[str, str | int | float | bool | None]] = Field(default_factory=dict)
    doctor: object | None = None
    warnings: list[str] = Field(default_factory=list)

class RunArtifact(BaseModel):
    artifact_name: str
    path: str

class RunManifest(BaseModel):
    manifest_version: str = RUN_MANIFEST_VERSION
    run_id: str
    generated_at: str
    input_inspection: InputInspection
    runtime: RuntimeModeSummary
    ontology_runtime: object | None = None
    record_counts: dict[str, int] = Field(default_factory=dict)
    stage_timings: dict[str, float] = Field(default_factory=dict)
    window_semantics: dict[str, str | None] = Field(default_factory=dict)
    report_versions: dict[str, str] = Field(default_factory=dict)
    output_artifacts: list[RunArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

def inspect_input_path(path, *, entry_sample_limit=12):
    if not path.exists():
        return InputInspection(input_path=str(path), input_kind='missing', warnings=[f'Input path does not exist: {path}'])
    if path.is_file() and path.suffix == '.jsonl':
        return InputInspection(input_path=str(path), input_kind='jsonl', jsonl_entry_count=1, total_container_entries=1, batch_entries=[InputBatchEntry(source_path=str(path), kind='jsonl')])
    if path.is_dir():
        entries = sorted(path.rglob('*.jsonl'))
        warnings = [] if entries else ['Directory input contains no .jsonl files']
        return InputInspection(input_path=str(path), input_kind='directory', jsonl_entry_count=len(entries), total_container_entries=len(entries), batch_entries=[InputBatchEntry(source_path=str(entry), kind='jsonl') for entry in entries[:entry_sample_limit]], warnings=warnings)
    if path.is_file() and path.suffix == '.zip':
        with ZipFile(path) as archive:
            names = sorted(archive.namelist())
        jsonl_entries = [name for name in names if name.endswith('.jsonl')]
        warnings = [] if jsonl_entries else [f'Zip archive contains no .jsonl files: {path.name}']
        return InputInspection(input_path=str(path), input_kind='zip', jsonl_entry_count=len(jsonl_entries), total_container_entries=len(names), batch_entries=[InputBatchEntry(source_path=f'{path}!{name}', kind='jsonl') for name in jsonl_entries[:entry_sample_limit]], warnings=warnings)
    return InputInspection(input_path=str(path), input_kind='unknown', warnings=[f'Unsupported input source: {path}'])

def summarize_runtime_mode(*, settings, enrichment, provider_runtime=None):
    embedding_provider_name = provider_runtime.embedding_provider.provenance.provider_name if provider_runtime is not None else settings.intelligence.embeddings.provider_kind
    topic_provider_name = provider_runtime.topic_provider.provenance.provider_name if provider_runtime is not None else settings.intelligence.topics.provider_kind
    language_provider = getattr(provider_runtime, 'language_id_provider', None)
    language_provider_name = language_provider.provenance.provider_name if language_provider is not None else settings.intelligence.language_id.fallback_provider_kind
    entity_mode = 'ml_enabled' if embedding_provider_name != 'token_overlap' or any(('phobert_ner' in method for fact in enrichment.entity_facts for method in fact.discovered_by)) or any((fact.matched_by in {'embedding_similarity', 'embedding_similarity_ranked'} for fact in enrichment.entity_facts)) else 'degraded_fallback'
    semantic_mode = 'ml_enabled' if any((component.name in {'semantic_assist_support', 'semantic_hypothesis_rerank', 'semantic_issue_rerank', 'semantic_corroboration_support'} for fact in [*enrichment.aspect_opinion_facts, *enrichment.issue_signal_facts] for component in fact.score_components)) else 'degraded_fallback'
    topic_mode = 'ml_enabled' if topic_provider_name == 'ontology_topic' or any(((artifact.provider_name or '').casefold() == 'ontology_topic' for artifact in enrichment.topic_artifacts)) else 'degraded_fallback'
    language_mode = 'ml_enabled' if language_provider_name == 'fasttext_lid' else 'degraded_fallback'
    requested_mode = 'default_intelligence' if settings.intelligence.enable_optional_ml_providers else 'degraded_fallback'
    effective_mode = 'default_intelligence' if requested_mode == 'default_intelligence' and any((mode == 'ml_enabled' for mode in (entity_mode, semantic_mode, topic_mode, language_mode))) else 'degraded_fallback'
    warnings = []
    if requested_mode == 'default_intelligence':
        degraded_components = [name for name, mode in {'entity': entity_mode, 'semantic': semantic_mode, 'topic': topic_mode, 'language_id': language_mode}.items() if mode == 'degraded_fallback']
        if degraded_components:
            warnings.append('Default intelligence was requested but some components degraded to fallback: ' + ', '.join(degraded_components))
    return RuntimeModeSummary(requested_mode=requested_mode, effective_mode=effective_mode, provider_modes={'entity': entity_mode, 'semantic': semantic_mode, 'topic': topic_mode, 'language_id': language_mode}, provider_provenance={'language_id': {'provider_kind': language_provider.provenance.provider_kind if language_provider is not None else settings.intelligence.language_id.provider_kind, 'provider_name': language_provider.provenance.provider_name if language_provider is not None else settings.intelligence.language_id.fallback_provider_kind, 'provider_version': language_provider.provenance.provider_version if language_provider is not None else 'unknown', 'model_id': language_provider.provenance.model_id if language_provider is not None else settings.intelligence.language_id.fasttext_model_path.name}, 'embedding': {'provider_kind': settings.intelligence.embeddings.provider_kind, 'provider_name': provider_runtime.embedding_provider.provenance.provider_name if provider_runtime is not None else settings.intelligence.embeddings.provider_kind, 'model_id': provider_runtime.embedding_provider.provenance.model_id if provider_runtime is not None else settings.intelligence.embeddings.model_id, 'device': provider_runtime.embedding_provider.provenance.device if provider_runtime is not None else settings.intelligence.embeddings.device}, 'vector_index': {'provider_kind': provider_runtime.vector_index.provenance.provider_name if provider_runtime is not None else settings.intelligence.vector_index.provider_kind, 'fallback_provider_kind': settings.intelligence.vector_index.fallback_provider_kind}, 'topic': {'provider_kind': provider_runtime.topic_provider.provenance.provider_name if provider_runtime is not None else settings.intelligence.topics.provider_kind, 'model_id': provider_runtime.topic_provider.provenance.model_id if provider_runtime is not None else settings.intelligence.embeddings.model_id, 'device': provider_runtime.topic_provider.provenance.device if provider_runtime is not None else None}, 'analytics': {'weighting_mode': settings.analytics.weighting_mode, 'dedup_enabled': settings.analytics.dedup.enabled, 'spam_enabled': settings.analytics.spam.enabled}}, doctor=run_runtime_doctor(settings), warnings=warnings)

def default_run_id():
    return utc_now().strftime('run-%Y%m%dT%H%M%SZ')

def build_run_manifest(*, run_id, input_path, settings, enrichment, bi_reports, storage_paths, ontology_runtime=None, input_inspection=None, record_counts=None, provider_runtime=None, stage_timings=None, extra_warnings=None, extra_notes=None):
    inspection = input_inspection or inspect_input_path(input_path)
    runtime = summarize_runtime_mode(settings=settings, enrichment=enrichment, provider_runtime=provider_runtime)
    warnings = [*inspection.warnings, *runtime.warnings, *(extra_warnings or [])]
    output_artifacts = [RunArtifact(artifact_name=name, path=path) for name, path in sorted(storage_paths.items(), key=lambda item: item[0])]
    return RunManifest(run_id=run_id, generated_at=utc_now().isoformat(), input_inspection=inspection, runtime=runtime, ontology_runtime=ontology_runtime.stack if ontology_runtime is not None else None, record_counts=record_counts or {}, stage_timings=stage_timings or {}, window_semantics={'analysis_start': bi_reports.sov_report.window.analysis_start, 'analysis_end': bi_reports.sov_report.window.analysis_end, 'delta_kind': bi_reports.sov_report.window.delta_kind, 'delta_comparison_mode': bi_reports.sov_report.window.delta_comparison_mode}, report_versions={'bi_schema_version': BI_SCHEMA_VERSION, 'sov_report_version': bi_reports.sov_report.contract.report_version, 'buzz_report_version': bi_reports.buzz_report.contract.report_version, 'emerging_topics_report_version': bi_reports.emerging_topics_report.contract.report_version, 'top_issues_report_version': bi_reports.top_issues_report.contract.report_version, 'thread_controversy_report_version': bi_reports.thread_controversy_report.contract.report_version, 'creator_source_breakdown_report_version': bi_reports.creator_source_breakdown_report.contract.report_version, 'insight_card_bundle_version': bi_reports.insight_card_bundle.contract.report_version}, output_artifacts=output_artifacts, warnings=warnings, notes=extra_notes or [])
