from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from smap.providers.base import ProviderMetadata, ProviderProvenance, VectorIndex, VectorItem, VectorNamespaceExpectation, VectorNamespaceInfo, VectorReuseState, VectorSearchHit
from smap.providers.errors import ProviderUnavailableError
from smap.providers.vector_index_manifest import build_manifest, load_manifest, namespace_dir, write_manifest
from smap.providers.vector_lifecycle import evaluate_manifest, manifest_info

@dataclass(slots=True)
class _NamespaceState:
    index: Any
    item_ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    metadata: list[dict[str, ProviderMetadata]] = field(default_factory=list)

class FaissVectorIndex(VectorIndex):

    def __init__(self, *, storage_dir=None):
        try:
            import faiss
        except ImportError as exc:
            raise ProviderUnavailableError('FAISS vector backend is unavailable.') from exc
        self._faiss = faiss
        self.version = 'faiss-vector-v2'
        self.provenance = ProviderProvenance(provider_kind='vector_index', provider_name='faiss', provider_version=self.version, model_id='IndexFlatIP', device='cpu')
        self._namespaces: dict[str, _NamespaceState] = {}
        self._expectations: dict[str, VectorNamespaceExpectation] = {}
        self.storage_dir = storage_dir
        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, *, namespace):
        self._namespaces.pop(namespace, None)
        if self.storage_dir is not None:
            for path in namespace_dir(self.storage_dir, namespace).glob('*'):
                path.unlink()
            namespace_dir(self.storage_dir, namespace).mkdir(parents=True, exist_ok=True)

    def bind_expectation(self, *, namespace, expected):
        self._expectations[namespace] = expected

    def expectation_for(self, *, namespace):
        return self._expectations.get(namespace)

    def load(self, *, namespace, expected=None, allow_stale=False, force=False):
        if self.storage_dir is None:
            return False
        resolved_expected = expected or self._expectations.get(namespace)
        if resolved_expected is None and (not allow_stale) and (not force):
            return False
        manifest = load_manifest(self.storage_dir, namespace)
        reuse_state, _ = evaluate_manifest(manifest, resolved_expected)
        if not force and reuse_state == VectorReuseState.STALE and (not allow_stale):
            return False
        if not force and reuse_state not in {VectorReuseState.VALID, VectorReuseState.STALE}:
            return False
        namespace_path = namespace_dir(self.storage_dir, namespace)
        index_path = namespace_path / 'index.faiss'
        payload_path = namespace_path / 'payload.json'
        if not index_path.exists() or not payload_path.exists():
            return False
        payload = json.loads(payload_path.read_text(encoding='utf-8'))
        namespace_state = _NamespaceState(index=self._faiss.read_index(str(index_path)))
        namespace_state.item_ids = [str(item_id) for item_id in payload.get('item_ids', [])]
        namespace_state.texts = [str(text) for text in payload.get('texts', [])]
        namespace_state.metadata = [dict(item) for item in payload.get('metadata', [])]
        self._namespaces[namespace] = namespace_state
        return True

    def upsert(self, items, *, namespace):
        if not items:
            return
        import numpy as np
        dimension = len(items[0].vector)
        state = _NamespaceState(index=self._faiss.IndexFlatIP(dimension))
        ordered = sorted(items, key=lambda item: item.item_id)
        matrix = np.array([item.vector for item in ordered], dtype='float32')
        state.index.add(matrix)
        state.item_ids = [item.item_id for item in ordered]
        state.texts = [item.text for item in ordered]
        state.metadata = [item.metadata for item in ordered]
        self._namespaces[namespace] = state
        if self.storage_dir is not None:
            namespace_path = namespace_dir(self.storage_dir, namespace)
            namespace_path.mkdir(parents=True, exist_ok=True)
            index_path = namespace_path / 'index.faiss'
            payload_path = namespace_path / 'payload.json'
            self._faiss.write_index(state.index, str(index_path))
            payload_path.write_text(json.dumps({'item_ids': state.item_ids, 'texts': state.texts, 'metadata': state.metadata}, ensure_ascii=False), encoding='utf-8')
            manifest = build_manifest(root=self.storage_dir, namespace=namespace, backend='faiss', items=ordered, storage_path=index_path, previous=load_manifest(self.storage_dir, namespace))
            write_manifest(self.storage_dir, manifest)

    def info(self, *, namespace, expected=None):
        if self.storage_dir is not None:
            manifest = load_manifest(self.storage_dir, namespace)
            if manifest is not None:
                return manifest_info(manifest, expected=expected)
        state = self._namespaces.get(namespace)
        if state is None or not state.item_ids:
            return None
        dimension = state.index.d if hasattr(state.index, 'd') else 0
        first_metadata = state.metadata[0] if state.metadata else {}
        return VectorNamespaceInfo(namespace=namespace, backend='faiss', item_count=len(state.item_ids), dimension=int(dimension), normalization_mode='cosine_normalized', embedding_model_id=str(first_metadata.get('embedding_model_id')) if first_metadata.get('embedding_model_id') is not None else None, embedding_provider_name=str(first_metadata.get('embedding_provider_name')) if first_metadata.get('embedding_provider_name') is not None else None, embedding_provider_version=str(first_metadata.get('embedding_provider_version')) if first_metadata.get('embedding_provider_version') is not None else None, storage_path=str(namespace_dir(self.storage_dir, namespace)) if self.storage_dir is not None else None, expected_corpus_hash=expected.corpus_hash if expected is not None else None)

    def search(self, vector, *, namespace, top_k=5):
        state = self._namespaces.get(namespace)
        if state is None:
            expected = self._expectations.get(namespace)
            if expected is None:
                return []
            if self.load(namespace=namespace, expected=expected):
                state = self._namespaces.get(namespace)
        if state is None:
            return []
        import numpy as np
        query = np.array([vector], dtype='float32')
        scores, indices = state.index.search(query, top_k)
        hits: list[VectorSearchHit] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if int(index) < 0 or int(index) >= len(state.item_ids):
                continue
            hits.append(VectorSearchHit(item_id=state.item_ids[int(index)], score=round(float(score), 6), text=state.texts[int(index)], metadata=state.metadata[int(index)]))
        return hits
