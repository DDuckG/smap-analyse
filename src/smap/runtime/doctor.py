from __future__ import annotations

import importlib.util
import sys

from pydantic import BaseModel, Field

from smap.core.settings import Settings
from smap.ontology.runtime import load_runtime_ontology
from smap.providers.errors import ProviderUnavailableError
from smap.providers.factory import (
    build_embedding_provider,
    build_language_id_provider,
    build_topic_provider,
    build_vector_index,
    language_id_model_candidates,
    resolve_language_id_model_path,
)


class DoctorComponentStatus(BaseModel):
    component: str
    requested: str | None = None
    selected: str | None = None
    available: bool = False
    model_available: bool = False
    degraded: bool = False
    device: str | None = None
    detail: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class RuntimeDoctorReport(BaseModel):
    cpu_or_gpu: str
    will_run: dict[str, DoctorComponentStatus] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _module_available(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except ValueError:
        return module_name in sys.modules


def run_runtime_doctor(settings: Settings) -> RuntimeDoctorReport:
    warnings: list[str] = []
    ontology = load_runtime_ontology(settings).registry
    language_provider = None
    embedding_provider = None
    topic_provider = None
    vector_index = None
    fasttext_module = _module_available("fasttext")
    resolved_model_path = resolve_language_id_model_path(settings)
    fasttext_model = resolved_model_path is not None
    model_detail = (
        str(resolved_model_path)
        if resolved_model_path is not None
        else "missing; searched " + ", ".join(str(path) for path in language_id_model_candidates(settings))
    )

    try:
        language_provider = build_language_id_provider(settings)
    except ProviderUnavailableError as exc:
        warnings.append(str(exc))

    try:
        embedding_provider = build_embedding_provider(settings)
    except ProviderUnavailableError as exc:
        warnings.append(str(exc))

    if embedding_provider is not None:
        try:
            topic_provider = build_topic_provider(
                settings,
                ontology=ontology,
                embedding_provider=embedding_provider,
            )
        except ProviderUnavailableError as exc:
            warnings.append(str(exc))

    try:
        vector_index = build_vector_index(settings)
    except ProviderUnavailableError as exc:
        warnings.append(str(exc))

    embedding_metadata: dict[str, object] = {
        "model_id": settings.intelligence.embeddings.model_id,
        "runtime_path": "onnx_cpu",
        "batch_size": settings.intelligence.embeddings.batch_size,
        "max_length": settings.intelligence.embeddings.max_length,
        "onnx_model_dir": str(settings.onnx_model_dir),
        "onnx_intra_op_threads": settings.intelligence.embeddings.onnx_intra_op_threads,
        "onnx_inter_op_threads": settings.intelligence.embeddings.onnx_inter_op_threads,
    }
    if embedding_provider is not None:
        embedding_metadata.update(dict(embedding_provider.provenance.run_metadata))

    embedding_detail = (
        f"{settings.intelligence.embeddings.model_id} "
        f"[backend=onnx, device=cpu, batch={settings.intelligence.embeddings.batch_size}, "
        f"max_length={settings.intelligence.embeddings.max_length}]"
    )

    report = RuntimeDoctorReport(
        cpu_or_gpu="cpu",
        will_run={
            "language_id": DoctorComponentStatus(
                component="language_id",
                requested=settings.intelligence.language_id.provider_kind,
                selected=language_provider.provenance.provider_name if language_provider is not None else None,
                available=fasttext_module and fasttext_model and language_provider is not None,
                model_available=fasttext_model,
                degraded=False,
                device=language_provider.provenance.device if language_provider is not None else None,
                detail=model_detail,
            ),
            "embeddings": DoctorComponentStatus(
                component="embeddings",
                requested="onnx_cpu",
                selected=embedding_provider.provenance.provider_name if embedding_provider is not None else None,
                available=embedding_provider is not None,
                model_available=embedding_provider is not None,
                degraded=False,
                device=embedding_provider.provenance.device if embedding_provider is not None else None,
                detail=embedding_detail,
                metadata=embedding_metadata,
            ),
            "topics": DoctorComponentStatus(
                component="topics",
                requested=settings.intelligence.topics.provider_kind,
                selected=topic_provider.provenance.provider_name if topic_provider is not None else None,
                available=topic_provider is not None,
                model_available=bool(ontology.topics),
                degraded=False,
                device=topic_provider.provenance.device if topic_provider is not None else None,
                detail=topic_provider.provenance.model_id if topic_provider is not None else None,
            ),
            "vector_index": DoctorComponentStatus(
                component="vector_index",
                requested=settings.intelligence.vector_index.provider_kind,
                selected=vector_index.provenance.provider_name if vector_index is not None else None,
                available=vector_index is not None,
                model_available=vector_index is not None,
                degraded=False,
                device=vector_index.provenance.device if vector_index is not None else None,
                detail=str(settings.vector_index_dir),
            ),
        },
        warnings=warnings,
    )

    seen: set[int] = set()
    for resource in (
        language_provider,
        embedding_provider,
        topic_provider,
        vector_index,
        getattr(embedding_provider, "cache_store", None),
    ):
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "close", None)
        if callable(close):
            close()
    return report
