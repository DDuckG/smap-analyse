from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import yaml

from smap.core.settings import Settings
from smap.ontology.models import OntologyOverlay, OntologyRegistry


def _read_yaml(path: Path) -> object:
    payload = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    if payload is None:
        raise ValueError(f"Ontology file is empty: {path}")
    return payload


def load_ontology_overlay(path: Path) -> OntologyOverlay:
    return OntologyOverlay.model_validate(_read_yaml(path))


def apply_ontology_overlays(
    registry: OntologyRegistry,
    overlays: Sequence[OntologyOverlay],
) -> OntologyRegistry:
    entity_by_id = {entity.id: entity for entity in registry.entities}
    topic_by_key = {topic.topic_key: topic for topic in registry.topics}
    aspect_by_id = {category.id: category for category in registry.aspect_categories}
    issue_by_id = {category.id: category for category in registry.issue_categories}
    alias_contributions = list(registry.alias_contributions)
    noise_terms = list(registry.noise_terms)
    loaded_overlays = list(registry.loaded_overlays)

    for overlay in overlays:
        loaded_overlays.append(overlay.metadata)
        for entity in overlay.entities:
            entity_by_id[entity.id] = entity
        for category in overlay.aspect_categories:
            aspect_by_id[category.id] = category
        for category in overlay.issue_categories:
            issue_by_id[category.id] = category
        for topic in overlay.topics:
            topic_by_key[topic.topic_key] = topic
        alias_contributions.extend(overlay.alias_contributions)
        noise_terms.extend(overlay.noise_terms)

    return OntologyRegistry(
        metadata=registry.metadata,
        domain_id=registry.domain_id,
        activation=registry.activation,
        entity_types=registry.entity_types,
        taxonomy_nodes=registry.taxonomy_nodes,
        aspect_categories=list(aspect_by_id.values()),
        intent_categories=registry.intent_categories,
        issue_categories=list(issue_by_id.values()),
        source_channels=registry.source_channels,
        entities=list(entity_by_id.values()),
        topics=list(topic_by_key.values()),
        alias_contributions=alias_contributions,
        noise_terms=noise_terms,
        loaded_overlays=loaded_overlays,
    )


def load_ontology(path: Path, overlay_paths: Sequence[Path] | None = None) -> OntologyRegistry:
    registry = OntologyRegistry.model_validate(_read_yaml(path))
    if not overlay_paths:
        return registry
    overlays = [load_ontology_overlay(overlay_path) for overlay_path in overlay_paths]
    return apply_ontology_overlays(registry, overlays)


def _review_overlay_paths(settings: Settings) -> list[Path]:
    if not settings.review_overlay_path.exists():
        return []
    return [settings.review_overlay_path]


def load_ontology_from_settings(settings: Settings) -> OntologyRegistry:
    return load_ontology(
        settings.selected_domain_ontology_path,
        overlay_paths=_review_overlay_paths(settings),
    )


def load_selected_domain_ontology(settings: Settings) -> OntologyRegistry:
    return load_ontology(
        settings.selected_domain_ontology_path,
        overlay_paths=_review_overlay_paths(settings),
    )
