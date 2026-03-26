from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path
from smap.providers.base import ProviderProvenance, VectorIndex, VectorItem, VectorNamespaceExpectation, VectorNamespaceInfo, VectorReuseState, VectorSearchHit
from smap.providers.vector_index_manifest import build_manifest, load_manifest, namespace_dir, write_manifest
from smap.providers.vector_lifecycle import evaluate_manifest, manifest_info

def _cosine_similarity(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum((a * b for a, b in zip(left, right, strict=True)))
    left_norm = math.sqrt(sum((a * a for a in left)))
    right_norm = math.sqrt(sum((b * b for b in right)))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)

class InMemoryVectorIndex(VectorIndex):

    def __init__(self, *, storage_dir=None):
        self.version = 'memory-vector-v2'
        self.provenance = ProviderProvenance(provider_kind='vector_index', provider_name='memory_exact', provider_version=self.version, model_id='exact-cosine', device='cpu')
        self._namespaces: dict[str, list[VectorItem]] = defaultdict(list)
        self._expectations: dict[str, VectorNamespaceExpectation] = {}
        self.storage_dir = storage_dir
        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, *, namespace):
        self._namespaces[namespace] = []
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
        state, _ = evaluate_manifest(manifest, resolved_expected)
        if not force and state == VectorReuseState.STALE and (not allow_stale):
            return False
        if not force and state not in {VectorReuseState.VALID, VectorReuseState.STALE}:
            return False
        items_path = namespace_dir(self.storage_dir, namespace) / 'items.json'
        if not items_path.exists():
            return False
        payload = json.loads(items_path.read_text(encoding='utf-8'))
        items = payload.get('items')
        if not isinstance(items, list):
            return False
        self._namespaces[namespace] = [VectorItem(item_id=str(item['item_id']), vector=tuple((float(value) for value in item['vector'])), text=str(item['text']), metadata=dict(item.get('metadata', {}))) for item in items]
        return True

    def upsert(self, items, *, namespace):
        if not items:
            return
        existing = {item.item_id: item for item in self._namespaces.get(namespace, [])}
        for item in items:
            existing[item.item_id] = item
        ordered = [existing[item_id] for item_id in sorted(existing)]
        self._namespaces[namespace] = ordered
        if self.storage_dir is not None:
            namespace_path = namespace_dir(self.storage_dir, namespace)
            namespace_path.mkdir(parents=True, exist_ok=True)
            items_path = namespace_path / 'items.json'
            items_path.write_text(json.dumps({'items': [{'item_id': item.item_id, 'vector': list(item.vector), 'text': item.text, 'metadata': item.metadata} for item in ordered]}, ensure_ascii=False), encoding='utf-8')
            manifest = build_manifest(root=self.storage_dir, namespace=namespace, backend='memory_exact', items=ordered, storage_path=items_path, previous=load_manifest(self.storage_dir, namespace))
            write_manifest(self.storage_dir, manifest)

    def info(self, *, namespace, expected=None):
        if namespace in self._namespaces and self._namespaces[namespace]:
            items = self._namespaces[namespace]
            if self.storage_dir is not None:
                manifest = load_manifest(self.storage_dir, namespace)
                if manifest is not None:
                    return manifest_info(manifest, expected=expected)
            first = items[0]
            return VectorNamespaceInfo(namespace=namespace, backend='memory_exact', item_count=len(items), dimension=len(first.vector), normalization_mode='cosine_normalized', embedding_model_id=str(first.metadata.get('embedding_model_id')) if first.metadata.get('embedding_model_id') is not None else None, embedding_provider_name=str(first.metadata.get('embedding_provider_name')) if first.metadata.get('embedding_provider_name') is not None else None, embedding_provider_version=str(first.metadata.get('embedding_provider_version')) if first.metadata.get('embedding_provider_version') is not None else None, storage_path=str(namespace_dir(self.storage_dir, namespace)) if self.storage_dir is not None else None, expected_corpus_hash=expected.corpus_hash if expected is not None else None)
        if self.storage_dir is None:
            return None
        manifest = load_manifest(self.storage_dir, namespace)
        return manifest_info(manifest, expected=expected)

    def search(self, vector, *, namespace, top_k=5):
        if namespace not in self._namespaces:
            expected = self._expectations.get(namespace)
            if expected is None:
                return []
            self.load(namespace=namespace, expected=expected)
        hits = [VectorSearchHit(item_id=item.item_id, score=round(_cosine_similarity(vector, item.vector), 6), text=item.text, metadata=item.metadata) for item in self._namespaces.get(namespace, [])]
        return sorted(hits, key=lambda item: (-item.score, item.item_id))[:top_k]
