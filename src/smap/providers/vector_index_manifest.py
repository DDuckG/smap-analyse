from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from smap.core.types import utc_now
from smap.providers.base import ProviderMetadata, VectorItem, VectorNamespaceInfo

class VectorNamespaceManifest(BaseModel):
    schema_version: str = 'vector-namespace-manifest-v1'
    namespace: str
    backend: str
    item_count: int
    dimension: int
    normalization_mode: str
    embedding_model_id: str | None = None
    embedding_provider_name: str | None = None
    embedding_provider_version: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    corpus_hash: str
    storage_path: str
    metadata: dict[str, ProviderMetadata] = Field(default_factory=dict)

    def as_info(self):
        return VectorNamespaceInfo(namespace=self.namespace, backend=self.backend, item_count=self.item_count, dimension=self.dimension, normalization_mode=self.normalization_mode, embedding_model_id=self.embedding_model_id, embedding_provider_name=self.embedding_provider_name, embedding_provider_version=self.embedding_provider_version, created_at=self.created_at, updated_at=self.updated_at, corpus_hash=self.corpus_hash, storage_path=self.storage_path, metadata=self.metadata)

def namespace_dir(root, namespace):
    return root / _safe_namespace(namespace)

def manifest_path(root, namespace):
    return namespace_dir(root, namespace) / 'manifest.json'

def build_manifest(*, root, namespace, backend, items, storage_path, previous=None):
    first_metadata = items[0].metadata if items else {}
    return VectorNamespaceManifest(namespace=namespace, backend=backend, item_count=len(items), dimension=len(items[0].vector) if items else 0, normalization_mode='cosine_normalized', embedding_model_id=_string_metadata(first_metadata, 'embedding_model_id'), embedding_provider_name=_string_metadata(first_metadata, 'embedding_provider_name'), embedding_provider_version=_string_metadata(first_metadata, 'embedding_provider_version'), created_at=previous.created_at if previous is not None else utc_now(), updated_at=utc_now(), corpus_hash=corpus_hash(items), storage_path=str(storage_path), metadata={'namespace': namespace, 'embedding_purpose': _string_metadata(first_metadata, 'embedding_purpose')})

def load_manifest(root, namespace):
    path = manifest_path(root, namespace)
    if not path.exists():
        return None
    return VectorNamespaceManifest.model_validate_json(path.read_text(encoding='utf-8'))

def write_manifest(root, manifest):
    path = manifest_path(root, manifest.namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding='utf-8')

def corpus_hash(items):
    payload = [{'item_id': item.item_id, 'text': item.text, 'vector_dim': len(item.vector), 'metadata': item.metadata} for item in sorted(items, key=lambda value: value.item_id)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()

def _safe_namespace(value):
    return value.replace('/', '__').replace('\\', '__').replace(':', '_')

def _string_metadata(metadata, key):
    value = metadata.get(key)
    return value if isinstance(value, str) else None
