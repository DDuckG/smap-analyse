from __future__ import annotations

from collections import defaultdict

from smap.ontology.models import OntologyRegistry, TaxonomyNode


def taxonomy_children(registry: OntologyRegistry) -> dict[str | None, list[TaxonomyNode]]:
    children: dict[str | None, list[TaxonomyNode]] = defaultdict(list)
    for node in registry.taxonomy_nodes:
        children[node.parent_id].append(node)
    return dict(children)


def taxonomy_path(registry: OntologyRegistry, node_id: str) -> list[TaxonomyNode]:
    by_id = {node.id: node for node in registry.taxonomy_nodes}
    path: list[TaxonomyNode] = []
    current = by_id[node_id]
    while True:
        path.append(current)
        if current.parent_id is None:
            break
        current = by_id[current.parent_id]
    path.reverse()
    return path

