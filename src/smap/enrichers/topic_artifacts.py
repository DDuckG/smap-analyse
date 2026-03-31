from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from smap.canonicalization.alias import normalize_alias
from smap.enrichers.models import (
    AspectOpinionFact,
    EntityFact,
    IssueSignalFact,
    TopicEvidencePhrase,
)
from smap.ontology.prototypes import PrototypeRegistry, TopicPrototypeBundle
from smap.providers.base import EmbeddingProvider, EmbeddingPurpose, TopicDocument

_NOISE_TOKENS = {
    "ban",
    "biet",
    "cho",
    "co",
    "cua",
    "haha",
    "hehe",
    "hello",
    "khong",
    "la",
    "nay",
    "nhe",
    "ok",
    "okay",
    "roi",
    "the",
    "thi",
    "this",
    "thoi",
    "va",
    "voi",
}
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,}", re.IGNORECASE)
VerifierLevel = Literal["a", "b", "c"]
TopicEvidenceRole = Literal["entity_anchor", "canonical", "supporting", "issue_support", "aspect_support", "debug_only"]


@dataclass(frozen=True, slots=True)
class TopicArtifactEvidence:
    entity_anchor_phrases: tuple[str, ...]
    canonical_evidence_phrases: tuple[str, ...]
    canonical_evidence_details: tuple[TopicEvidencePhrase, ...]
    supporting_phrases: tuple[str, ...]
    supporting_phrase_details: tuple[TopicEvidencePhrase, ...]
    issue_supporting_phrases: tuple[str, ...]
    aspect_supporting_phrases: tuple[str, ...]
    top_canonical_entity_ids: tuple[str, ...]
    aspect_profile: tuple[str, ...]
    issue_profile: tuple[str, ...]
    artifact_purity_score: float
    evidence_coherence_score: float
    issue_leak_rate: float


def build_topic_artifact_evidence(
    *,
    topic_bundle: TopicPrototypeBundle | None,
    documents: list[TopicDocument],
    entity_facts: list[EntityFact],
    aspect_facts: list[AspectOpinionFact],
    issue_facts: list[IssueSignalFact],
    prototype_registry: PrototypeRegistry | None,
    embedding_provider: EmbeddingProvider | None,
    text_embedding_cache: dict[str, tuple[float, ...]] | None = None,
    artifact_level: VerifierLevel = "b",
) -> TopicArtifactEvidence:
    mention_ids = {document.mention_id for document in documents}
    entity_counts = Counter(
        fact.canonical_entity_id
        for fact in entity_facts
        if fact.canonical_entity_id is not None and fact.mention_id in mention_ids
    )
    aspect_counts = Counter(
        fact.aspect
        for fact in aspect_facts
        if fact.target_kind in {"canonical_entity", "concept"} and fact.mention_id in mention_ids
    )
    issue_counts = Counter(
        fact.issue_category
        for fact in issue_facts
        if fact.target_kind in {"canonical_entity", "concept"} and fact.mention_id in mention_ids
    )

    prototype_vector = topic_bundle.vector if topic_bundle is not None else None
    issue_centric = topic_bundle.issue_centric if topic_bundle is not None else False
    entity_labels = {
        entity_id: prototype_registry.entities[entity_id].label
        for entity_id in entity_counts
        if prototype_registry is not None and entity_id in prototype_registry.entities
    }
    normalized_entity_labels = {
        normalize_alias(label)
        for label in entity_labels.values()
        if normalize_alias(label)
    }
    phrase_support_counts: Counter[str] = Counter()
    phrase_sources: dict[str, set[str]] = defaultdict(set)

    for entity_id, label in entity_labels.items():
        normalized = normalize_alias(label)
        if normalized:
            phrase_support_counts[normalized] += entity_counts[entity_id]
            phrase_sources[normalized].add("entity")
    for aspect_id, count in aspect_counts.items():
        normalized = normalize_alias(aspect_id.replace("_", " "))
        if normalized:
            phrase_support_counts[normalized] += count
            phrase_sources[normalized].add("aspect")
    for issue_id, count in issue_counts.items():
        normalized = normalize_alias(issue_id.replace("_", " "))
        if normalized:
            phrase_support_counts[normalized] += count
            phrase_sources[normalized].add("issue")
    if topic_bundle is not None:
        for seed in topic_bundle.seed_phrases:
            normalized = normalize_alias(seed)
            if normalized:
                phrase_support_counts[normalized] += 2
                phrase_sources[normalized].add("seed")

    recurring_phrase_counts = Counter(
        phrase
        for document in documents
        for phrase in _candidate_phrases_from_document(document.normalized_text)
    )
    for phrase, count in recurring_phrase_counts.items():
        if count < 1:
            continue
        phrase_support_counts[phrase] += count
        phrase_sources[phrase].add("document")

    candidate_phrases = list(phrase_support_counts)
    phrase_embeddings: dict[str, tuple[float, ...]] = {}
    if embedding_provider is not None and candidate_phrases:
        phrase_embeddings = _lookup_embeddings(
            candidate_phrases,
            embedding_provider=embedding_provider,
            purpose=EmbeddingPurpose.CLUSTERING,
            text_embedding_cache=text_embedding_cache,
        )

    scored_details: list[tuple[TopicEvidencePhrase, float]] = []
    for phrase in candidate_phrases:
        hygiene = _phrase_hygiene(phrase)
        readability = _phrase_readability(phrase)
        if hygiene < 0.45 or readability < 0.4:
            continue
        support = min(phrase_support_counts[phrase] / max(len(documents), 1), 1.0)
        sources = phrase_sources[phrase]
        entity_bonus = 0.18 if "entity" in sources else 0.0
        aspect_bonus = 0.1 if "aspect" in sources else 0.0
        seed_bonus = 0.08 if "seed" in sources else 0.0
        issue_bonus = 0.04 if "issue" in sources else 0.0
        document_bonus = 0.03 if "document" in sources and len(sources) > 1 else 0.0
        semantic = 0.0
        if prototype_vector is not None and phrase in phrase_embeddings:
            semantic = max(
                sum(a * b for a, b in zip(phrase_embeddings[phrase], prototype_vector, strict=True)),
                0.0,
            )
        aspect_alignment = _prototype_alignment(
            phrase,
            prototype_ids=(topic_bundle.related_aspect_ids if topic_bundle is not None else ()),
            registry_items=(prototype_registry.aspects if prototype_registry is not None else {}),
            phrase_embeddings=phrase_embeddings,
        )
        issue_alignment = _prototype_alignment(
            phrase,
            prototype_ids=(topic_bundle.related_issue_ids if topic_bundle is not None else ()),
            registry_items=(prototype_registry.issues if prototype_registry is not None else {}),
            phrase_embeddings=phrase_embeddings,
        )
        entity_anchor_candidate = (
            "entity" in sources
            and "seed" not in sources
            and not ("aspect" in sources or "issue" in sources)
            and (phrase in normalized_entity_labels or len(sources) == 1)
        )
        document_only = "document" in sources and len(sources) == 1
        issue_contamination = (
            "issue" in sources
            and "entity" not in sources
            and "aspect" not in sources
            and "seed" not in sources
            and not issue_centric
        )
        role_features = _artifact_role_features(
            hygiene=hygiene,
            readability=readability,
            support=support,
            semantic=semantic,
            aspect_alignment=aspect_alignment,
            issue_alignment=issue_alignment,
            sources=sources,
            phrase_support_count=phrase_support_counts[phrase],
            issue_contamination=issue_contamination,
            issue_centric=issue_centric,
            entity_anchor_candidate=entity_anchor_candidate,
        )
        verifier_distribution = _artifact_role_distribution(role_features, artifact_level=artifact_level)
        canonical_probability = verifier_distribution.get("canonical", 0.0)
        supporting_probability = verifier_distribution.get("supporting", 0.0)
        issue_probability = verifier_distribution.get("issue_support", 0.0)
        aspect_probability = verifier_distribution.get("aspect_support", 0.0)
        purity = (
            hygiene * 0.32
            + readability * 0.26
            + semantic * 0.22
            + support * 0.1
            + entity_bonus
            + aspect_bonus
            + seed_bonus
            + document_bonus
            + canonical_probability * 0.12
            + supporting_probability * 0.04
        )
        if entity_anchor_candidate and semantic < 0.72:
            purity -= 0.1
        if issue_contamination:
            purity -= max(0.24, issue_probability * 0.28)
        if document_only:
            purity -= 0.12
        purity = round(max(min(purity, 0.99), 0.0), 6)
        total_score = round(
            max(
                min(
                    hygiene * 0.24
                    + readability * 0.18
                    + support * 0.18
                    + semantic * 0.2
                    + entity_bonus
                    + aspect_bonus
                    + seed_bonus
                    + issue_bonus
                    + document_bonus
                    + (canonical_probability * 0.1)
                    + (aspect_probability * 0.03)
                    - max(0.14 if issue_contamination else 0.0, issue_probability * 0.14),
                    0.99,
                ),
                0.0,
            ),
            6,
        )
        role: TopicEvidenceRole = _phrase_role(
            sources=sources,
            purity=purity,
            total_score=total_score,
            semantic=semantic,
            aspect_alignment=aspect_alignment,
            issue_alignment=issue_alignment,
            entity_anchor_candidate=entity_anchor_candidate,
            issue_centric=issue_centric,
            artifact_level=artifact_level,
        )
        provenance_flags = sorted(
            {
                *(f"{source}_support" for source in sorted(sources)),
                "recurrent_phrase" if phrase_support_counts[phrase] >= 2 else "",
                "issue_contamination_guard" if issue_contamination else "",
                f"verifier_role:{role}",
                "entity_anchor_candidate" if entity_anchor_candidate else "",
            }
            - {""}
        )
        scored_details.append(
            (
                TopicEvidencePhrase(
                    phrase=_display_phrase(phrase, entity_labels, prototype_registry),
                    role=role,
                    source_types=sorted(sources),
                    purity_score=round(purity, 4),
                    readability_score=round(readability, 4),
                    semantic_score=round(semantic, 4) if semantic > 0.0 else None,
                    support_score=round(support, 4),
                    provenance_flags=provenance_flags,
                ),
                total_score,
            )
        )

    scored_details.sort(
        key=lambda item: (
            -item[0].purity_score,
            -item[1],
            item[0].role,
            item[0].phrase,
        )
    )
    entity_anchor_details = _select_diverse_details(
        [detail for detail, _ in scored_details if detail.role == "entity_anchor"],
        limit=4,
    )
    canonical_details = _select_coherent_details(
        [detail for detail, _ in scored_details if detail.role == "canonical"],
        phrase_embeddings=phrase_embeddings,
        prototype_vector=prototype_vector,
        limit=5,
    )
    supporting_details = _select_diverse_details(
        [detail for detail, _ in scored_details if detail.role == "supporting"],
        limit=5,
    )
    issue_details = _select_diverse_details(
        [detail for detail, _ in scored_details if detail.role == "issue_support"],
        limit=4,
    )
    aspect_details = _select_diverse_details(
        [detail for detail, _ in scored_details if detail.role == "aspect_support"],
        limit=4,
    )
    fallback_pool = [
        detail
        for detail in supporting_details
        if detail.purity_score >= 0.64 and detail.readability_score >= 0.58
    ]
    if issue_centric and not fallback_pool:
        fallback_pool = [
            detail
            for detail in supporting_details
            if detail.purity_score >= 0.48 and detail.readability_score >= 0.42
        ]
    if issue_centric and not fallback_pool:
        fallback_pool = [
            detail
            for detail in issue_details
            if detail.purity_score >= 0.52 and detail.readability_score >= 0.48
        ]
    business_fallback = fallback_pool[:2] if not canonical_details else []
    final_canonical = tuple(canonical_details or business_fallback)
    purity_basis = final_canonical or tuple(
        detail
        for detail in supporting_details
        if detail.purity_score >= 0.54 and detail.readability_score >= 0.5
    )[:2]
    coherence_score = _evidence_coherence_score(
        purity_basis,
        phrase_embeddings=phrase_embeddings,
        prototype_vector=prototype_vector,
    )
    artifact_purity_score = (
        round(
            (
                sum(detail.purity_score for detail in purity_basis) / max(len(purity_basis), 1)
            ) * 0.82
            + coherence_score * 0.18,
            4,
        )
        if purity_basis
        else 0.0
    )
    issue_leak_rate = round(
        (
            len([detail for detail in final_canonical if detail.role == "issue_support"])
            / max(len(final_canonical), 1)
        )
        if final_canonical
        else 0.0,
        4,
    )
    return TopicArtifactEvidence(
        entity_anchor_phrases=tuple(detail.phrase for detail in entity_anchor_details),
        canonical_evidence_phrases=tuple(detail.phrase for detail in final_canonical),
        canonical_evidence_details=tuple(final_canonical),
        supporting_phrases=tuple(detail.phrase for detail in supporting_details),
        supporting_phrase_details=(*supporting_details, *issue_details, *aspect_details),
        issue_supporting_phrases=tuple(detail.phrase for detail in issue_details),
        aspect_supporting_phrases=tuple(detail.phrase for detail in aspect_details),
        top_canonical_entity_ids=tuple(entity_id for entity_id, _ in entity_counts.most_common(5)),
        aspect_profile=tuple(aspect_id for aspect_id, _ in aspect_counts.most_common(5)),
        issue_profile=tuple(issue_id for issue_id, _ in issue_counts.most_common(5)),
        artifact_purity_score=artifact_purity_score,
        evidence_coherence_score=coherence_score,
        issue_leak_rate=issue_leak_rate,
    )


def _phrase_role(
    *,
    sources: set[str],
    purity: float,
    total_score: float,
    semantic: float,
    aspect_alignment: float,
    issue_alignment: float,
    entity_anchor_candidate: bool,
    issue_centric: bool,
    artifact_level: VerifierLevel,
) -> TopicEvidenceRole:
    issue_only = "issue" in sources and "entity" not in sources and "aspect" not in sources and "seed" not in sources
    if entity_anchor_candidate and semantic < 0.82:
        return "entity_anchor"
    if artifact_level in {"b", "c"} and issue_alignment >= max(semantic + 0.04, aspect_alignment + 0.06, 0.62):
        return "issue_support"
    if artifact_level in {"b", "c"} and aspect_alignment >= max(issue_alignment + 0.02, 0.58):
        if purity >= 0.62 and total_score >= 0.56 and semantic >= 0.64:
            return "canonical"
        return "aspect_support"
    if issue_centric and "issue" in sources and purity >= 0.54 and total_score >= 0.52:
        return "canonical"
    if issue_only and not issue_centric:
        return "issue_support"
    if (
        purity >= 0.62
        and total_score >= 0.58
        and (
            "entity" in sources
            or "aspect" in sources
            or "seed" in sources
            or ("document" in sources and len(sources) > 1 and semantic >= 0.78)
            or semantic >= 0.76
            or (issue_centric and "issue" in sources and total_score >= 0.64)
        )
    ):
        return "canonical"
    if "aspect" in sources and "entity" not in sources:
        return "aspect_support"
    if purity >= 0.48 and total_score >= 0.44:
        return "supporting"
    return "debug_only"


def _artifact_role_features(
    *,
    hygiene: float,
    readability: float,
    support: float,
    semantic: float,
    aspect_alignment: float,
    issue_alignment: float,
    sources: set[str],
    phrase_support_count: int,
    issue_contamination: bool,
    issue_centric: bool,
    entity_anchor_candidate: bool,
) -> dict[str, float]:
    return {
        "hygiene": hygiene,
        "readability": readability,
        "support": support,
        "semantic": semantic,
        "aspect_alignment": aspect_alignment,
        "issue_alignment": issue_alignment,
        "entity_support": 1.0 if "entity" in sources else 0.0,
        "aspect_support": 1.0 if "aspect" in sources else 0.0,
        "issue_support": 1.0 if "issue" in sources else 0.0,
        "seed_support": 1.0 if "seed" in sources else 0.0,
        "document_support": 1.0 if "document" in sources else 0.0,
        "source_diversity": min(float(len(sources)) / 4.0, 1.0),
        "recurrence": min(float(phrase_support_count) / 4.0, 1.0),
        "issue_contamination": 1.0 if issue_contamination else 0.0,
        "issue_centric_topic": 1.0 if issue_centric else 0.0,
        "document_only": 1.0 if "document" in sources and len(sources) == 1 else 0.0,
        "entity_anchor_candidate": 1.0 if entity_anchor_candidate else 0.0,
    }


def _artifact_role_distribution(
    features: dict[str, float],
    *,
    artifact_level: VerifierLevel,
) -> dict[TopicEvidenceRole, float]:
    entity_anchor = min(
        1.0,
        features["entity_anchor_candidate"] * 0.78
        + features["entity_support"] * 0.14
        + max(0.0, 0.62 - features["semantic"]) * 0.12,
    )
    issue_support = min(
        1.0,
        features["issue_support"] * 0.38
        + features["issue_alignment"] * 0.44
        + features["issue_contamination"] * 0.18,
    )
    aspect_support = min(
        1.0,
        features["aspect_support"] * 0.34
        + features["aspect_alignment"] * 0.42
        + features["semantic"] * 0.14,
    )
    canonical = min(
        1.0,
        features["semantic"] * 0.36
        + features["readability"] * 0.18
        + features["hygiene"] * 0.16
        + features["support"] * 0.1
        + features["seed_support"] * 0.08
        + features["aspect_alignment"] * 0.08
        + max(features["source_diversity"] - features["issue_alignment"], 0.0) * 0.06
        - features["entity_anchor_candidate"] * 0.14
        - features["issue_contamination"] * 0.18,
    )
    supporting = min(
        1.0,
        features["readability"] * 0.24
        + features["support"] * 0.2
        + features["semantic"] * 0.18
        + features["source_diversity"] * 0.12
        + features["document_support"] * 0.08
        - features["document_only"] * 0.12,
    )
    if artifact_level == "c":
        canonical = min(1.0, canonical + max(features["semantic"] - 0.72, 0.0) * 0.14)
        supporting = max(0.0, supporting - features["issue_alignment"] * 0.08)
        issue_support = min(1.0, issue_support + max(features["issue_alignment"] - 0.66, 0.0) * 0.12)
    return {
        "entity_anchor": round(max(min(entity_anchor, 1.0), 0.0), 6),
        "canonical": round(max(min(canonical, 1.0), 0.0), 6),
        "supporting": round(max(min(supporting, 1.0), 0.0), 6),
        "issue_support": round(max(min(issue_support, 1.0), 0.0), 6),
        "aspect_support": round(max(min(aspect_support, 1.0), 0.0), 6),
        "debug_only": 0.0,
    }


def _prototype_alignment(
    phrase: str,
    *,
    prototype_ids: tuple[str, ...],
    registry_items: Mapping[str, object],
    phrase_embeddings: dict[str, tuple[float, ...]],
) -> float:
    phrase_key = normalize_alias(phrase)
    phrase_vector = phrase_embeddings.get(phrase_key)
    if phrase_vector is None or not prototype_ids:
        return 0.0
    scores: list[float] = []
    for prototype_id in prototype_ids:
        bundle = registry_items.get(prototype_id)
        vector = getattr(bundle, "vector", None)
        if vector is None:
            continue
        scores.append(max(sum(a * b for a, b in zip(phrase_vector, vector, strict=True)), 0.0))
    if not scores:
        return 0.0
    return round(max(scores), 4)


def _candidate_phrases_from_document(text: str) -> Iterable[str]:
    tokens = [
        token
        for token in _TOKEN_RE.findall(text)
        if len(token) >= 2 and token.casefold() not in _NOISE_TOKENS
    ]
    token_count = len(tokens)
    for size in (3, 2):
        if token_count < size:
            continue
        for index in range(token_count - size + 1):
            phrase_tokens = tokens[index : index + size]
            phrase = normalize_alias(" ".join(phrase_tokens))
            if phrase and _phrase_hygiene(phrase) >= 0.52:
                yield phrase


def _phrase_hygiene(phrase: str) -> float:
    normalized = normalize_alias(phrase)
    if not normalized:
        return 0.0
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return 0.0
    if any(token.startswith("#") or token.startswith("@") for token in tokens):
        return 0.0
    if len(tokens) == 1 and tokens[0] in _NOISE_TOKENS:
        return 0.0
    informative_tokens = sum(
        1
        for token in tokens
        if token not in _NOISE_TOKENS and (len(token) >= 4 or any(character.isdigit() for character in token))
    )
    score = 0.34
    if informative_tokens:
        score += 0.28
    if len(tokens) >= 2:
        score += 0.18
    if any(character.isdigit() for character in normalized):
        score += 0.08
    if any(token in _NOISE_TOKENS for token in tokens):
        score -= 0.12
    return round(max(min(score, 1.0), 0.0), 4)


def _phrase_readability(phrase: str) -> float:
    normalized = normalize_alias(phrase)
    if not normalized:
        return 0.0
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return 0.0
    score = 0.34
    if len(tokens) >= 2:
        score += 0.2
    if any(token in _NOISE_TOKENS for token in tokens):
        score -= 0.18
    if any(token.startswith("#") or token.startswith("@") or "http" in token for token in tokens):
        score -= 0.3
    if len(tokens) == 1 and len(tokens[0]) < 4 and not any(character.isdigit() for character in tokens[0]):
        score -= 0.14
    return round(max(min(score, 1.0), 0.0), 4)


def _select_diverse_details(details: list[TopicEvidencePhrase], *, limit: int) -> list[TopicEvidencePhrase]:
    selected: list[TopicEvidencePhrase] = []
    selected_tokens: list[set[str]] = []
    for detail in details:
        phrase_tokens = set(normalize_alias(detail.phrase).split())
        if any(_jaccard(phrase_tokens, existing) >= 0.8 for existing in selected_tokens):
            continue
        selected.append(detail)
        selected_tokens.append(phrase_tokens)
        if len(selected) >= limit:
            break
    return selected


def _select_coherent_details(
    details: list[TopicEvidencePhrase],
    *,
    phrase_embeddings: dict[str, tuple[float, ...]],
    prototype_vector: tuple[float, ...] | None,
    limit: int,
) -> list[TopicEvidencePhrase]:
    remaining = list(details)
    selected: list[TopicEvidencePhrase] = []
    selected_tokens: list[set[str]] = []
    while remaining and len(selected) < limit:
        best_detail: TopicEvidencePhrase | None = None
        best_score = -1.0
        for detail in remaining:
            phrase_tokens = set(normalize_alias(detail.phrase).split())
            if any(_jaccard(phrase_tokens, existing) >= 0.8 for existing in selected_tokens):
                continue
            coherence = _detail_coherence(
                detail,
                selected,
                phrase_embeddings=phrase_embeddings,
                prototype_vector=prototype_vector,
            )
            source_priority = (
                0.12
                if "entity" in detail.source_types
                else 0.09
                if "aspect" in detail.source_types
                else 0.07
                if "seed" in detail.source_types
                else 0.02
                if "document" in detail.source_types and len(detail.source_types) > 1
                else -0.04
            )
            score = (
                detail.purity_score * 0.58
                + detail.readability_score * 0.14
                + coherence * 0.18
                + source_priority
            )
            if score > best_score:
                best_score = score
                best_detail = detail
        if best_detail is None:
            break
        selected.append(best_detail)
        selected_tokens.append(set(normalize_alias(best_detail.phrase).split()))
        remaining = [detail for detail in remaining if detail != best_detail]
    return selected


def _display_phrase(
    phrase: str,
    entity_labels: dict[str, str],
    prototype_registry: PrototypeRegistry | None,
) -> str:
    if prototype_registry is None:
        return phrase.replace("_", " ").strip()
    for label in entity_labels.values():
        if normalize_alias(label) == phrase:
            return label
    return phrase.replace("_", " ").strip()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _detail_coherence(
    detail: TopicEvidencePhrase,
    selected: list[TopicEvidencePhrase],
    *,
    phrase_embeddings: dict[str, tuple[float, ...]],
    prototype_vector: tuple[float, ...] | None,
) -> float:
    phrase_key = normalize_alias(detail.phrase)
    coherence_scores: list[float] = []
    current_vector = phrase_embeddings.get(phrase_key)
    if prototype_vector is not None and current_vector is not None:
        coherence_scores.append(
            max(sum(a * b for a, b in zip(current_vector, prototype_vector, strict=True)), 0.0)
        )
    for item in selected:
        other_key = normalize_alias(item.phrase)
        lexical = _jaccard(set(phrase_key.split()), set(other_key.split()))
        other_vector = phrase_embeddings.get(other_key)
        if current_vector is not None and other_vector is not None:
            semantic = max(sum(a * b for a, b in zip(current_vector, other_vector, strict=True)), 0.0)
            coherence_scores.append((lexical * 0.28) + (semantic * 0.72))
        else:
            coherence_scores.append(lexical)
    if not coherence_scores:
        return 0.0
    return round(sum(coherence_scores) / len(coherence_scores), 4)


def _evidence_coherence_score(
    details: tuple[TopicEvidencePhrase, ...],
    *,
    phrase_embeddings: dict[str, tuple[float, ...]],
    prototype_vector: tuple[float, ...] | None,
) -> float:
    if not details:
        return 0.0
    scores = [
        _detail_coherence(
            detail,
            [item for item in details if item != detail],
            phrase_embeddings=phrase_embeddings,
            prototype_vector=prototype_vector,
        )
        for detail in details
    ]
    return round(sum(scores) / len(scores), 4)


def _lookup_embeddings(
    texts: list[str],
    *,
    embedding_provider: EmbeddingProvider,
    purpose: EmbeddingPurpose,
    text_embedding_cache: dict[str, tuple[float, ...]] | None = None,
) -> dict[str, tuple[float, ...]]:
    cache = text_embedding_cache if text_embedding_cache is not None else {}
    missing = [text for text in dict.fromkeys(texts) if text and text not in cache]
    if missing:
        vectors = embedding_provider.embed_texts(missing, purpose=purpose)
        cache.update(zip(missing, vectors, strict=True))
    return {
        text: cache[text]
        for text in texts
        if text in cache
    }
