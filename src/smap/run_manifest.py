from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from zipfile import ZipFile

from pydantic import BaseModel, Field

from smap.bi.contracts import BI_SCHEMA_VERSION
from smap.bi.models import BIReportBundle
from smap.core.settings import Settings
from smap.core.types import utc_now
from smap.enrichers.models import EnrichmentBundle
from smap.ingestion.readers import iter_jsonl_archive_members
from smap.ontology.runtime import OntologyRuntime, OntologyRuntimeStack
from smap.providers.factory import ProviderRuntime
from smap.runtime.doctor import RuntimeDoctorReport, run_runtime_doctor

RUN_MANIFEST_VERSION = "2026.04.08-onnx-cpu-r1"


class InputBatchEntry(BaseModel):
    source_path: str
    kind: Literal["jsonl"]


class InputInspection(BaseModel):
    input_path: str
    input_kind: Literal["jsonl", "zip", "directory", "missing", "unknown"]
    jsonl_entry_count: int = 0
    total_container_entries: int | None = None
    batch_entries: list[InputBatchEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RuntimeModeSummary(BaseModel):
    requested_mode: Literal["onnx_cpu_intelligence"]
    effective_mode: Literal["onnx_cpu_intelligence"]
    provider_modes: dict[str, str] = Field(default_factory=dict)
    provider_provenance: dict[str, dict[str, str | int | float | bool | None]] = Field(default_factory=dict)
    doctor: RuntimeDoctorReport | None = None
    warnings: list[str] = Field(default_factory=list)


class RunArtifact(BaseModel):
    artifact_name: str
    path: str


class SampleInputSelection(BaseModel):
    source_archive_path: str
    staged_input_path: str
    selection_mode: Literal["zip_record_sample"]
    max_records: int
    selected_record_count: int
    selected_jsonl_entry_count: int
    total_jsonl_entry_count: int
    truncated_final_entry: bool = False
    selected_entries: list[InputBatchEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RunManifest(BaseModel):
    manifest_version: str = RUN_MANIFEST_VERSION
    release_name: str = "R1"
    run_id: str
    generated_at: str
    input_inspection: InputInspection
    sample_input_selection: SampleInputSelection | None = None
    runtime: RuntimeModeSummary
    ontology_runtime: OntologyRuntimeStack | None = None
    record_counts: dict[str, int] = Field(default_factory=dict)
    stage_timings: dict[str, float] = Field(default_factory=dict)
    window_semantics: dict[str, str | None] = Field(default_factory=dict)
    report_versions: dict[str, str] = Field(default_factory=dict)
    qa_suites_run: list[str] = Field(default_factory=list)
    output_artifacts: list[RunArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ReleaseSummary(BaseModel):
    release_name: str = "R1"
    run_id: str
    generated_at: str
    input_path: str
    sample_input_selection: SampleInputSelection | None = None
    requested_mode: str
    effective_mode: str
    outputs_generated: list[RunArtifact] = Field(default_factory=list)
    qa_artifacts: list[RunArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


def inspect_input_path(path: Path, *, entry_sample_limit: int = 12) -> InputInspection:
    if not path.exists():
        return InputInspection(
            input_path=str(path),
            input_kind="missing",
            warnings=[f"Input path does not exist: {path}"],
        )
    if path.is_file() and path.suffix == ".jsonl":
        return InputInspection(
            input_path=str(path),
            input_kind="jsonl",
            jsonl_entry_count=1,
            total_container_entries=1,
            batch_entries=[InputBatchEntry(source_path=str(path), kind="jsonl")],
        )
    if path.is_dir():
        entries = sorted(path.rglob("*.jsonl"))
        warnings: list[str] = []
        if not entries:
            warnings.append("Directory input contains no .jsonl files")
        return InputInspection(
            input_path=str(path),
            input_kind="directory",
            jsonl_entry_count=len(entries),
            total_container_entries=len(entries),
            batch_entries=[
                InputBatchEntry(source_path=str(entry), kind="jsonl")
                for entry in entries[:entry_sample_limit]
            ],
            warnings=warnings,
        )
    if path.is_file() and path.suffix == ".zip":
        with ZipFile(path) as archive:
            names = sorted(archive.namelist())
            jsonl_entries = list(iter_jsonl_archive_members(archive))
        warnings = [] if jsonl_entries else [f"Zip archive contains no .jsonl files: {path.name}"]
        return InputInspection(
            input_path=str(path),
            input_kind="zip",
            jsonl_entry_count=len(jsonl_entries),
            total_container_entries=len(names),
            batch_entries=[
                InputBatchEntry(source_path=f"{path}!{name}", kind="jsonl")
                for name in jsonl_entries[:entry_sample_limit]
            ],
            warnings=warnings,
        )
    return InputInspection(
        input_path=str(path),
        input_kind="unknown",
        warnings=[f"Unsupported input source: {path}"],
    )


def summarize_runtime_mode(
    *,
    settings: Settings,
    enrichment: EnrichmentBundle,
    provider_runtime: ProviderRuntime | None = None,
) -> RuntimeModeSummary:
    embedding_provider_name = (
        provider_runtime.embedding_provider.provenance.provider_name
        if provider_runtime is not None
        else "phobert_embedding"
    )
    topic_provider_name = (
        provider_runtime.topic_provider.provenance.provider_name
        if provider_runtime is not None
        else settings.intelligence.topics.provider_kind
    )
    language_provider = getattr(provider_runtime, "language_id_provider", None)
    language_provider_name = (
        language_provider.provenance.provider_name
        if language_provider is not None
        else settings.intelligence.language_id.provider_kind
    )

    entity_mode = (
        "enabled"
        if embedding_provider_name == "phobert_embedding"
        or any("phobert_ner" in method for fact in enrichment.entity_facts for method in fact.discovered_by)
        or any(
            fact.matched_by in {"embedding_similarity", "embedding_similarity_ranked"}
            for fact in enrichment.entity_facts
        )
        else "missing"
    )
    semantic_mode = "enabled" if embedding_provider_name == "phobert_embedding" else "missing"
    topic_mode = (
        "enabled"
        if topic_provider_name == "ontology_topic"
        or any((artifact.provider_name or "").casefold() == "ontology_topic" for artifact in enrichment.topic_artifacts)
        else "missing"
    )
    language_mode = "enabled" if language_provider_name == "fasttext_lid" else "missing"

    warnings = [
        f"Runtime component `{name}` was not recorded as enabled."
        for name, mode in {
            "entity": entity_mode,
            "semantic": semantic_mode,
            "topic": topic_mode,
            "language_id": language_mode,
        }.items()
        if mode != "enabled"
    ]

    return RuntimeModeSummary(
        requested_mode="onnx_cpu_intelligence",
        effective_mode="onnx_cpu_intelligence",
        provider_modes={
            "entity": entity_mode,
            "semantic": semantic_mode,
            "topic": topic_mode,
            "language_id": language_mode,
        },
        provider_provenance={
            "language_id": {
                "provider_kind": (
                    language_provider.provenance.provider_kind
                    if language_provider is not None
                    else settings.intelligence.language_id.provider_kind
                ),
                "provider_name": (
                    language_provider.provenance.provider_name
                    if language_provider is not None
                    else settings.intelligence.language_id.provider_kind
                ),
                "provider_version": (
                    language_provider.provenance.provider_version
                    if language_provider is not None
                    else "unknown"
                ),
                "model_id": (
                    language_provider.provenance.model_id
                    if language_provider is not None
                    else settings.intelligence.language_id.fasttext_model_path.name
                ),
            },
            "embedding": {
                "provider_kind": settings.intelligence.embeddings.provider_kind,
                "model_id": (
                    provider_runtime.embedding_provider.provenance.model_id
                    if provider_runtime is not None
                    else settings.intelligence.embeddings.model_id
                ),
                "device": (
                    provider_runtime.embedding_provider.provenance.device
                    if provider_runtime is not None
                    else settings.intelligence.embeddings.device
                ),
                "provider_name": (
                    provider_runtime.embedding_provider.provenance.provider_name
                    if provider_runtime is not None
                    else settings.intelligence.embeddings.provider_kind
                ),
                "backend": (
                    provider_runtime.embedding_provider.provenance.run_metadata.get("backend")
                    if provider_runtime is not None
                    else settings.intelligence.embeddings.runtime_backend
                ),
            },
            "vector_index": {
                "provider_kind": (
                    provider_runtime.vector_index.provenance.provider_name
                    if provider_runtime is not None
                    else settings.intelligence.vector_index.provider_kind
                ),
            },
            "ner": {
                "provider_kind": settings.intelligence.ner.provider_kind,
                "provider_name": "phobert_ner" if entity_mode == "enabled" else "missing",
                "model_id": settings.intelligence.ner.model_id,
            },
            "topic": {
                "provider_kind": (
                    provider_runtime.topic_provider.provenance.provider_name
                    if provider_runtime is not None
                    else settings.intelligence.topics.provider_kind
                ),
                "enabled": settings.intelligence.topics.enabled,
                "model_id": (
                    provider_runtime.topic_provider.provenance.model_id
                    if provider_runtime is not None
                    else settings.intelligence.embeddings.model_id
                ),
                "device": (
                    provider_runtime.topic_provider.provenance.device
                    if provider_runtime is not None
                    else None
                ),
            },
            "analytics": {
                "weighting_mode": settings.analytics.weighting_mode,
                "dedup_enabled": settings.analytics.dedup.enabled,
                "spam_enabled": settings.analytics.spam.enabled,
            },
        },
        doctor=run_runtime_doctor(settings),
        warnings=warnings,
    )


def default_run_id() -> str:
    return utc_now().strftime("run-%Y%m%dT%H%M%SZ")


def build_run_manifest(
    *,
    run_id: str,
    input_path: Path,
    settings: Settings,
    enrichment: EnrichmentBundle,
    bi_reports: BIReportBundle,
    storage_paths: dict[str, str],
    ontology_runtime: OntologyRuntime | None = None,
    input_inspection: InputInspection | None = None,
    record_counts: dict[str, int] | None = None,
    provider_runtime: ProviderRuntime | None = None,
    stage_timings: dict[str, float] | None = None,
    qa_suites_run: list[str] | None = None,
    sample_input_selection: SampleInputSelection | None = None,
    extra_warnings: list[str] | None = None,
    extra_notes: list[str] | None = None,
) -> RunManifest:
    inspection = input_inspection or inspect_input_path(input_path)
    runtime = summarize_runtime_mode(
        settings=settings,
        enrichment=enrichment,
        provider_runtime=provider_runtime,
    )
    warnings = [*inspection.warnings, *runtime.warnings, *(extra_warnings or [])]
    output_artifacts = [
        RunArtifact(artifact_name=name, path=path)
        for name, path in sorted(storage_paths.items(), key=lambda item: item[0])
    ]
    return RunManifest(
        run_id=run_id,
        generated_at=utc_now().isoformat(),
        input_inspection=inspection,
        sample_input_selection=sample_input_selection,
        runtime=runtime,
        ontology_runtime=ontology_runtime.stack if ontology_runtime is not None else None,
        record_counts=record_counts or {},
        stage_timings=stage_timings or {},
        window_semantics={
            "analysis_start": bi_reports.sov_report.window.analysis_start,
            "analysis_end": bi_reports.sov_report.window.analysis_end,
            "delta_kind": bi_reports.sov_report.window.delta_kind,
            "delta_comparison_mode": bi_reports.sov_report.window.delta_comparison_mode,
        },
        report_versions={
            "bi_schema_version": BI_SCHEMA_VERSION,
            "sov_report_version": bi_reports.sov_report.contract.report_version,
            "buzz_report_version": bi_reports.buzz_report.contract.report_version,
            "emerging_topics_report_version": bi_reports.emerging_topics_report.contract.report_version,
            "top_issues_report_version": bi_reports.top_issues_report.contract.report_version,
            "thread_controversy_report_version": bi_reports.thread_controversy_report.contract.report_version,
            "creator_source_breakdown_report_version": bi_reports.creator_source_breakdown_report.contract.report_version,
            "insight_card_bundle_version": bi_reports.insight_card_bundle.contract.report_version,
        },
        qa_suites_run=qa_suites_run or [],
        output_artifacts=output_artifacts,
        warnings=warnings,
        notes=extra_notes or [],
    )


def build_release_summary(
    *,
    run_manifest: RunManifest,
    output_artifacts: list[RunArtifact],
    qa_artifacts: list[RunArtifact],
    sample_input_selection: SampleInputSelection | None = None,
    warnings: list[str] | None = None,
    skipped: list[str] | None = None,
) -> ReleaseSummary:
    return ReleaseSummary(
        run_id=run_manifest.run_id,
        generated_at=utc_now().isoformat(),
        input_path=run_manifest.input_inspection.input_path,
        sample_input_selection=sample_input_selection,
        requested_mode=run_manifest.runtime.requested_mode,
        effective_mode=run_manifest.runtime.effective_mode,
        outputs_generated=output_artifacts,
        qa_artifacts=qa_artifacts,
        warnings=[*run_manifest.warnings, *(warnings or [])],
        skipped=skipped or [],
    )


def write_model_json(path: Path, payload: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        return
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
