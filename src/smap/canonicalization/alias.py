from __future__ import annotations
import re
from collections.abc import Iterable
from dataclasses import dataclass
from smap.canonicalization.models import CanonicalEntity, EntityAlias
from smap.ontology.models import OntologyRegistry
from smap.review.applicability_engine import ApplicabilityEngine
from smap.review.context import ReviewContext
NORMALIZE_RE = re.compile('[^\\w\\s]', flags=re.UNICODE)

def normalize_alias(text):
    repaired = _repair_common_mojibake(text)
    normalized = NORMALIZE_RE.sub(' ', repaired.lower())
    return ' '.join(normalized.split())

def _repair_common_mojibake(text):
    if not any((marker in text for marker in ('Ã', 'Æ', 'â', 'ð'))):
        return text
    try:
        repaired = text.encode('latin1').decode('utf-8')
    except UnicodeError:
        return text
    suspicious_markers = ('Ã', 'Æ', 'â', 'ð')
    before = sum((text.count(marker) for marker in suspicious_markers))
    after = sum((repaired.count(marker) for marker in suspicious_markers))
    if repaired and after < before:
        return repaired
    return text

@dataclass(slots=True)
class AliasRegistry:
    entities: dict[str, CanonicalEntity]
    aliases_by_exact: dict[str, list[EntityAlias]]
    aliases_by_normalized: dict[str, list[EntityAlias]]
    noise_terms: set[str]

    @classmethod
    def from_ontology(cls, registry, *, review_context=None):
        applicability_engine = ApplicabilityEngine()
        entities: dict[str, CanonicalEntity] = {}
        aliases_by_exact: dict[str, list[EntityAlias]] = {}
        aliases_by_normalized: dict[str, list[EntityAlias]] = {}
        for entity_seed in registry.entities:
            entity = CanonicalEntity(canonical_entity_id=entity_seed.id, name=entity_seed.name, entity_type=entity_seed.entity_type, entity_kind=entity_seed.entity_kind, knowledge_layer=entity_seed.knowledge_layer, active_linking=entity_seed.active_linking, target_eligible=entity_seed.target_eligible, taxonomy_ids=entity_seed.taxonomy_ids, description=entity_seed.description, anti_confusion_phrases=entity_seed.anti_confusion_phrases)
            entities[entity.canonical_entity_id] = entity
            if not entity_seed.active_linking:
                continue
            for alias_value in [entity_seed.name, *entity_seed.aliases, *entity_seed.compact_aliases]:
                cls._register_alias(aliases_by_exact=aliases_by_exact, aliases_by_normalized=aliases_by_normalized, alias=EntityAlias(alias_id=f'{entity.canonical_entity_id}:{normalize_alias(alias_value)}', canonical_entity_id=entity.canonical_entity_id, alias=alias_value, normalized_alias=normalize_alias(alias_value)))
        for contribution in registry.alias_contributions:
            if not applicability_engine.alias_contribution_applies(contribution, review_context):
                continue
            cls._register_alias(aliases_by_exact=aliases_by_exact, aliases_by_normalized=aliases_by_normalized, alias=EntityAlias(alias_id=f'{contribution.canonical_entity_id}:{normalize_alias(contribution.alias)}:{contribution.source}', canonical_entity_id=contribution.canonical_entity_id, alias=contribution.alias, normalized_alias=normalize_alias(contribution.alias), source=contribution.source))
        return cls(entities=entities, aliases_by_exact=aliases_by_exact, aliases_by_normalized=aliases_by_normalized, noise_terms={normalize_alias(item.term) for item in registry.noise_terms if applicability_engine.noise_term_applies(item, review_context)})

    @staticmethod
    def _register_alias(*, aliases_by_exact, aliases_by_normalized, alias):
        exact_key = alias.alias.casefold()
        normalized_key = alias.normalized_alias
        aliases_by_exact.setdefault(exact_key, []).append(alias)
        aliases_by_normalized.setdefault(normalized_key, []).append(alias)

    def candidate_strings(self):
        return {entity_id: entity.name for entity_id, entity in self.entities.items()}

    def vector_alias_strings(self):
        return {alias.alias_id: (alias.canonical_entity_id, alias.alias, alias.source) for alias in self.all_aliases()}

    def all_aliases(self):
        seen: set[str] = set()
        for aliases in self.aliases_by_normalized.values():
            for alias in aliases:
                if alias.alias_id in seen:
                    continue
                seen.add(alias.alias_id)
                yield alias

    def find_exact_aliases(self, text):
        return list(self.aliases_by_exact.get(text.casefold(), []))

    def find_normalized_aliases(self, text):
        return list(self.aliases_by_normalized.get(normalize_alias(text), []))

    def is_noise_term(self, text):
        return normalize_alias(text) in self.noise_terms

def boundary_contains(text, needle):
    if not text or not needle:
        return False
    pattern = re.compile(f'(?<!\\w){re.escape(needle)}(?!\\w)', flags=re.IGNORECASE | re.UNICODE)
    return pattern.search(text) is not None

def boundary_spans(text, needle):
    if not text or not needle:
        return []
    pattern = re.compile(f'(?<!\\w){re.escape(needle)}(?!\\w)', flags=re.IGNORECASE | re.UNICODE)
    return [(match.start(), match.end()) for match in pattern.finditer(text)]
