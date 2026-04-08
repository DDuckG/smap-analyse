from __future__ import annotations

from dataclasses import asdict

from smap.providers.base import VectorNamespaceExpectation, VectorNamespaceInfo, VectorReuseState
from smap.providers.vector_index_manifest import VectorNamespaceManifest


def evaluate_manifest(
    manifest: VectorNamespaceManifest | None,
    expected: VectorNamespaceExpectation | None,
) -> tuple[VectorReuseState, list[str]]:
    if manifest is None:
        return VectorReuseState.MISSING, ["manifest_missing"]
    if expected is None:
        return VectorReuseState.VALID, []
    errors: list[str] = []
    stale_reasons: list[str] = []
    if expected.namespace != manifest.namespace:
        errors.append("namespace_mismatch")
    if expected.backend is not None and expected.backend != manifest.backend:
        errors.append("backend_mismatch")
    if expected.dimension is not None and expected.dimension != manifest.dimension:
        errors.append("dimension_mismatch")
    if expected.normalization_mode is not None and expected.normalization_mode != manifest.normalization_mode:
        errors.append("normalization_mode_mismatch")
    if expected.embedding_model_id is not None and expected.embedding_model_id != manifest.embedding_model_id:
        errors.append("embedding_model_id_mismatch")
    if expected.embedding_provider_name is not None and expected.embedding_provider_name != manifest.embedding_provider_name:
        errors.append("embedding_provider_name_mismatch")
    if (
        expected.embedding_provider_version is not None
        and expected.embedding_provider_version != manifest.embedding_provider_version
    ):
        errors.append("embedding_provider_version_mismatch")
    manifest_purpose = manifest.metadata.get("embedding_purpose")
    if expected.embedding_purpose is not None and expected.embedding_purpose != manifest_purpose:
        errors.append("embedding_purpose_mismatch")
    if expected.corpus_hash is not None:
        if not manifest.corpus_hash:
            return VectorReuseState.REFRESH_REQUIRED, ["corpus_hash_missing"]
        if expected.corpus_hash != manifest.corpus_hash:
            stale_reasons.append("corpus_hash_mismatch")
    if errors:
        return VectorReuseState.INCOMPATIBLE, errors
    if stale_reasons:
        return VectorReuseState.STALE, stale_reasons
    return VectorReuseState.VALID, []


def manifest_info(
    manifest: VectorNamespaceManifest | None,
    *,
    expected: VectorNamespaceExpectation | None,
) -> VectorNamespaceInfo | None:
    if manifest is None and expected is None:
        return None
    if manifest is None and expected is not None:
        return VectorNamespaceInfo(
            namespace=expected.namespace,
            backend=expected.backend or "unknown",
            item_count=0,
            dimension=expected.dimension or 0,
            normalization_mode=expected.normalization_mode or "unknown",
            embedding_model_id=expected.embedding_model_id,
            embedding_provider_name=expected.embedding_provider_name,
            embedding_provider_version=expected.embedding_provider_version,
            expected_corpus_hash=expected.corpus_hash,
            reuse_state=VectorReuseState.MISSING.value,
            compatibility_errors=["manifest_missing"],
            recommended_action="vector-index-build",
        )
    assert manifest is not None
    state, errors = evaluate_manifest(manifest, expected)
    payload = asdict(manifest.as_info())
    payload["expected_corpus_hash"] = expected.corpus_hash if expected is not None else None
    payload["reuse_state"] = state.value
    payload["compatibility_errors"] = errors
    payload["recommended_action"] = _recommended_action(state)
    return VectorNamespaceInfo(**payload)


def _recommended_action(state: VectorReuseState) -> str | None:
    if state == VectorReuseState.MISSING:
        return "vector-index-build"
    if state == VectorReuseState.STALE:
        return "vector-index-refresh"
    if state in {VectorReuseState.INCOMPATIBLE, VectorReuseState.REFRESH_REQUIRED}:
        return "vector-index-refresh --force"
    return None
