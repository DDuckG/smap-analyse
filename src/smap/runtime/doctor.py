import importlib.util
import sys
from pydantic import BaseModel, Field
from smap.ontology.runtime import load_runtime_ontology
from smap.providers.factory import build_embedding_provider, build_language_id_provider, build_topic_provider, build_vector_index, language_id_model_candidates, resolve_language_id_model_path

class DoctorComponentStatus(BaseModel):
    component: str
    requested: str | None = None
    selected: str | None = None
    available: bool = False
    model_available: bool = False
    degraded: bool = False
    device: str | None = None
    detail: str | None = None

class RuntimeDoctorReport(BaseModel):
    cpu_or_gpu: str
    will_run: dict[str, DoctorComponentStatus] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

def _module_available(module_name):
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except ValueError:
        return module_name in sys.modules

def _runtime_device():
    try:
        import torch
    except ImportError:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return 'mps'
    return 'cpu'

def run_runtime_doctor(settings):
    warnings = []
    ontology = load_runtime_ontology(settings).registry
    language_provider = build_language_id_provider(settings)
    embedding_provider = build_embedding_provider(settings)
    topic_provider = build_topic_provider(settings, ontology=ontology, embedding_provider=embedding_provider)
    vector_index = build_vector_index(settings)
    fasttext_requested = settings.intelligence.language_id.provider_kind == 'fasttext'
    fasttext_module = _module_available('fasttext')
    resolved_model_path = resolve_language_id_model_path(settings)
    fasttext_model = resolved_model_path is not None
    model_detail = str(resolved_model_path) if resolved_model_path is not None else 'missing; searched ' + ', '.join((str(path) for path in language_id_model_candidates(settings)))
    if fasttext_requested and language_provider.provenance.provider_name != 'fasttext_lid':
        warnings.append('fastText was requested for LID but runtime will use heuristic fallback. ' + model_detail)
    report = RuntimeDoctorReport(cpu_or_gpu=_runtime_device(), will_run={'language_id': DoctorComponentStatus(component='language_id', requested=settings.intelligence.language_id.provider_kind, selected=language_provider.provenance.provider_name, available=fasttext_module if fasttext_requested else True, model_available=fasttext_model, degraded=language_provider.provenance.provider_name != 'fasttext_lid' if fasttext_requested else False, device=language_provider.provenance.device, detail=model_detail), 'embeddings': DoctorComponentStatus(component='embeddings', requested=settings.intelligence.embeddings.provider_kind, selected=embedding_provider.provenance.provider_name, available=_module_available('transformers') and _module_available('torch'), model_available=True, degraded=embedding_provider.provenance.provider_name == 'token_overlap', device=embedding_provider.provenance.device, detail=embedding_provider.provenance.model_id), 'topics': DoctorComponentStatus(component='topics', requested=settings.intelligence.topics.provider_kind, selected=topic_provider.provenance.provider_name, available=bool(ontology.topics), model_available=True, degraded=topic_provider.provenance.provider_name != 'ontology_topic', device=topic_provider.provenance.device, detail=topic_provider.provenance.model_id), 'vector_index': DoctorComponentStatus(component='vector_index', requested=settings.intelligence.vector_index.provider_kind, selected=vector_index.provenance.provider_name, available=_module_available('faiss'), model_available=True, degraded=vector_index.provenance.provider_name != 'faiss', device=vector_index.provenance.device, detail=str(settings.vector_index_dir))}, warnings=warnings)
    for resource in (embedding_provider, getattr(embedding_provider, 'cache_store', None), vector_index):
        close_method = getattr(resource, 'close', None)
        if callable(close_method):
            close_method()
    return report
