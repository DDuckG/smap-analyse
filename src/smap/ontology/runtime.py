import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from smap.canonicalization.alias import AliasRegistry, normalize_alias
from smap.contracts.uap import content_keywords, content_title
from smap.ontology.loader import load_ontology
DomainSelectionMode = Literal['explicit', 'corpus_auto', 'fallback_default']

class OntologyRuntimeStack(BaseModel):
    ontology_path: str
    ontology_name: str
    ontology_version: str
    ontology_fingerprint: str
    domain_id: str | None = None
    selection_mode: DomainSelectionMode
    selection_reason: str
    selection_score: float | None = None
    matched_record_count: int | None = None
    matched_terms: list[str] = Field(default_factory=list)

@dataclass(frozen=True, slots=True)
class OntologyRuntime:
    registry: object
    stack: OntologyRuntimeStack
    other_domain_aliases: frozenset[str]

@dataclass(frozen=True, slots=True)
class _DomainScore:
    path: Path
    registry: object
    score: float
    matched_record_count: int
    matched_terms: tuple[str, ...]

def load_runtime_ontology(settings, *, records=None):
    explicit_path = _explicit_domain_path(settings)
    if explicit_path is not None:
        registry = load_ontology(explicit_path)
        stack = _runtime_stack(registry=registry, path=explicit_path, selection_mode='explicit', selection_reason='explicit_domain_selection')
        return OntologyRuntime(registry=registry, stack=stack, other_domain_aliases=frozenset())
    candidate_paths = settings.available_domain_ontology_paths()
    if records and candidate_paths:
        scored = [_score_domain(path, records) for path in candidate_paths]
        best = _best_domain_score(scored)
        if best is not None:
            stack = _runtime_stack(registry=best.registry, path=best.path, selection_mode='corpus_auto', selection_reason='domain_corpus_match', selection_score=best.score, matched_record_count=best.matched_record_count, matched_terms=list(best.matched_terms))
            return OntologyRuntime(registry=best.registry, stack=stack, other_domain_aliases=frozenset(_collect_other_domain_aliases(scored, selected_path=best.path)))
    fallback_path = _fallback_domain_path(settings)
    registry = load_ontology(fallback_path)
    stack = _runtime_stack(registry=registry, path=fallback_path, selection_mode='fallback_default', selection_reason='default_domain_fallback')
    return OntologyRuntime(registry=registry, stack=stack, other_domain_aliases=frozenset())

def _explicit_domain_path(settings):
    configured = settings.domain_ontology_path
    if configured is None:
        return None
    return configured.resolve()

def _fallback_domain_path(settings):
    candidate_paths = settings.available_domain_ontology_paths()
    if settings.domain_ontology_path is not None and settings.domain_ontology_path.exists():
        return settings.domain_ontology_path.resolve()
    preferred = ['cosmetics_vn.yaml', 'beer_vn.yaml', 'blockchain_vn.yaml']
    for name in preferred:
        candidate = settings.domain_ontology_dir / name
        if candidate.exists():
            return candidate.resolve()
    if candidate_paths:
        return candidate_paths[0]
    raise FileNotFoundError(f'No domain ontology YAML files found in {settings.domain_ontology_dir}')

def _runtime_stack(*, registry, path, selection_mode, selection_reason, selection_score=None, matched_record_count=None, matched_terms=None):
    return OntologyRuntimeStack(ontology_path=str(path), ontology_name=registry.metadata.name, ontology_version=registry.metadata.version, ontology_fingerprint=_fingerprint_path(path), domain_id=registry.domain_id, selection_mode=selection_mode, selection_reason=selection_reason, selection_score=selection_score, matched_record_count=matched_record_count, matched_terms=matched_terms or [])

def _fingerprint_path(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

def _score_domain(path, records):
    registry = load_ontology(path)
    weighted_terms = _domain_scoring_terms(registry)
    if not records or not weighted_terms:
        return _DomainScore(path=path, registry=registry, score=0.0, matched_record_count=0, matched_terms=())
    total_score = 0.0
    matched_record_count = 0
    matched_terms = {}
    for record in records:
        text = _record_activation_text(record)
        if not text:
            continue
        record_hits = {}
        for term, weight in weighted_terms:
            if term in text:
                record_hits[term] = max(record_hits.get(term, 0.0), weight)
        if not record_hits:
            continue
        matched_record_count += 1
        total_score += sum(record_hits.values())
        for term, weight in record_hits.items():
            matched_terms[term] = max(matched_terms.get(term, 0.0), weight)
    ranked_terms = tuple((term for term, _ in sorted(matched_terms.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[:8]))
    return _DomainScore(path=path, registry=registry, score=round(total_score, 4), matched_record_count=matched_record_count, matched_terms=ranked_terms)

def _best_domain_score(scored):
    if not scored:
        return None
    ranked = sorted(scored, key=lambda item: (-item.score, -item.matched_record_count, item.path.name))
    best = ranked[0]
    if best.score <= 0.0 or best.matched_record_count <= 0:
        return None
    return best

def _domain_scoring_terms(registry):
    weighted_terms = {}
    if registry.activation is not None:
        for signal in registry.activation.signals:
            normalized = normalize_alias(signal.phrase)
            if len(normalized) >= 3:
                weighted_terms[normalized] = max(weighted_terms.get(normalized, 0.0), signal.weight)
    for entity in registry.entities:
        for alias in [entity.name, *entity.aliases, *entity.compact_aliases]:
            normalized = normalize_alias(alias)
            if len(normalized) < 3:
                continue
            weight = 1.3 if any((character.isdigit() for character in normalized)) else 1.0
            weighted_terms[normalized] = max(weighted_terms.get(normalized, 0.0), weight)
    for topic in registry.topics:
        for phrase in [topic.label, *topic.seed_phrases]:
            normalized = normalize_alias(phrase)
            if len(normalized) < 4:
                continue
            weight = 0.9 if ' ' in normalized else 0.6
            weighted_terms[normalized] = max(weighted_terms.get(normalized, 0.0), weight)
    return sorted(weighted_terms.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))

def _collect_other_domain_aliases(scored, *, selected_path):
    inactive = set()
    for item in scored:
        if item.path == selected_path:
            continue
        inactive.update(_registry_alias_terms(item.registry))
    return inactive

def _registry_alias_terms(registry):
    alias_terms = set()
    for entity in registry.entities:
        alias_terms.update((normalize_alias(alias) for alias in [entity.name, *entity.aliases, *entity.compact_aliases] if normalize_alias(alias)))
    registry_aliases = AliasRegistry.from_ontology(registry)
    alias_terms.update((alias.normalized_alias for alias in registry_aliases.all_aliases()))
    return {term for term in alias_terms if len(term) >= 3}

def _record_activation_text(record):
    hashtags = getattr(record.content, 'hashtags', None) or []
    keywords = content_keywords(record)
    summary_title = content_title(record) or ''
    parts = [record.text, summary_title, *[str(hashtag) for hashtag in hashtags if hashtag], *[str(keyword) for keyword in keywords if keyword]]
    return normalize_alias(' '.join((part for part in parts if part)))
