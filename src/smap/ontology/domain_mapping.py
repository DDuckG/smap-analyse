from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import yaml

LEGACY_OVERLAY_TO_DOMAIN_FILENAME = {
    "acme_social.yaml": "acme_social.yaml",
    "ambiguous_nova.yaml": "ambiguous_nova.yaml",
    "domain_ev_manufacturing.yaml": "ev_vn.yaml",
    "domain_facial_cleanser_vn.yaml": "facial_cleanser_vn.yaml",
    "example_beer.yaml": "beer_vn.yaml",
    "example_cosmetics.yaml": "cosmetics_example_vn.yaml",
    "glowlabs_pack.yaml": "glowlabs_pack.yaml",
    "glowlabs_pack_variant.yaml": "glowlabs_pack_variant.yaml",
}

_EQUIVALENT_DOMAIN_FILENAME_GROUPS = (
    frozenset({"cosmetics_vn.yaml", "facial_cleanser_vn.yaml"}),
)

_REVIEW_OVERLAY_FILENAMES = {
    "reviewed_aliases.yaml",
    "reviewed_aliases.candidate.yaml",
}


def legacy_overlay_domain_candidates(
    overlay_paths: Sequence[Path],
    domain_dir: Path,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for overlay_path in overlay_paths:
        filename = LEGACY_OVERLAY_TO_DOMAIN_FILENAME.get(overlay_path.name.casefold())
        if filename is None:
            continue
        candidate = (domain_dir / filename).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def legacy_overlay_domain_path(
    overlay_paths: Sequence[Path],
    domain_dir: Path,
) -> Path | None:
    candidates = [candidate for candidate in legacy_overlay_domain_candidates(overlay_paths, domain_dir) if candidate.exists()]
    if len(candidates) != 1:
        return None
    return candidates[0]


def equivalent_domain_filenames(filename: str) -> frozenset[str]:
    normalized = filename.casefold()
    for group in _EQUIVALENT_DOMAIN_FILENAME_GROUPS:
        if normalized in group:
            return group
    return frozenset({normalized})


def is_review_overlay_file(path: Path) -> bool:
    if path.name.casefold() in _REVIEW_OVERLAY_FILENAMES:
        return True
    if not path.exists() or path.suffix.casefold() not in {".yaml", ".yml"}:
        return False
    try:
        payload = cast(dict[str, Any] | None, yaml.safe_load(path.read_text(encoding="utf-8")))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False
    source = str(metadata.get("source", "")).casefold()
    layer_kind = str(metadata.get("layer_kind", "")).casefold()
    return source == "review" or layer_kind == "review_overlay"
