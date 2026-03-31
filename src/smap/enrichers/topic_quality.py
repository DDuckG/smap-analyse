from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from smap.enrichers.models import TopicArtifactFact
from smap.hil.topic_lineage import topic_lineage_matches
from smap.providers.base import EmbeddingProvider, EmbeddingPurpose, TopicArtifact, TopicDocument
from smap.storage.repository import write_jsonl

_GENERIC_TOPIC_TOKENS = {
    "ban",
    "biet",
    "cho",
    "co",
    "di",
    "duoc",
    "for",
    "good",
    "haha",
    "hehe",
    "hello",
    "khong",
    "la",
    "minh",
    "nha",
    "nay",
    "nhe",
    "nhung",
    "ok",
    "okay",
    "one",
    "on",
    "qua",
    "roi",
    "the",
    "thi",
    "this",
    "thoi",
    "va",
}


def build_topic_term_signature(top_terms: Sequence[str]) -> str:
    canonical_terms = sorted({term.casefold().strip() for term in top_terms if term.strip()})
    return hashlib.sha256(json.dumps(canonical_terms, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_topic_profile_signature(
    *,
    topic_key: str,
    topic_label: str,
    topic_family: str | None = None,
    top_canonical_entity_ids: Sequence[str],
    entity_anchor_phrases: Sequence[str] = (),
    aspect_profile: Sequence[str],
    issue_profile: Sequence[str],
    canonical_evidence_phrases: Sequence[str],
) -> str:
    payload = {
        "topic_key": topic_key.casefold().strip(),
        "topic_label": topic_label.casefold().strip(),
        "topic_family": (topic_family or topic_key).casefold().strip(),
        "top_canonical_entity_ids": sorted({item for item in top_canonical_entity_ids if item}),
        "entity_anchor_phrases": sorted(
            {
                phrase.casefold().strip()
                for phrase in entity_anchor_phrases
                if phrase.strip()
            }
        ),
        "aspect_profile": sorted({item for item in aspect_profile if item}),
        "issue_profile": sorted({item for item in issue_profile if item}),
        "canonical_evidence_phrases": sorted(
            {
                phrase.casefold().strip()
                for phrase in canonical_evidence_phrases
                if phrase.strip()
            }
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_topic_signature(
    *,
    provider_name: str,
    model_id: str,
    top_terms: Sequence[str],
    representative_document_ids: Sequence[str],
) -> str:
    payload = {
        "provider_name": provider_name,
        "model_id": model_id,
        "term_signature": build_topic_term_signature(top_terms),
        "representatives": sorted(representative_document_ids),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_topic_run_id(
    *,
    provider_name: str,
    model_id: str,
    document_ids: Sequence[str],
) -> str:
    payload = {
        "provider_name": provider_name,
        "model_id": model_id,
        "document_ids": sorted(document_ids),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class TopicQualityAssessment:
    quality_score: float
    usefulness_score: float
    reportability_score: float
    reporting_status: str
    weak_topic: bool
    noisy_topic: bool
    hashtag_only: bool
    chatter_burden: float
    reaction_only_burden: float
    low_information_burden: float
    evidence_density: float
    recurrence_score: float
    business_salience_score: float
    salient_terms: tuple[str, ...]
    reason_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TopicLabelChoice:
    reporting_topic_key: str
    reporting_topic_label: str
    label_source: str
    label_confidence: float


@dataclass(frozen=True, slots=True)
class TopicLabelHealth:
    score: float
    flags: tuple[str, ...]


def assess_topic_artifact(
    artifact: TopicArtifact,
    documents: Sequence[TopicDocument],
    *,
    canonical_evidence_phrases: Sequence[str] | None = None,
    supporting_phrases: Sequence[str] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    global_document_frequency: dict[str, int] | None = None,
    text_embedding_cache: dict[str, tuple[float, ...]] | None = None,
) -> TopicQualityAssessment:
    topic_size = max(artifact.topic_size, 0)
    top_terms = [term for term in artifact.top_terms if term.strip()]
    canonical_evidence = tuple(item for item in (canonical_evidence_phrases or []) if item.strip())
    supporting_evidence = tuple(item for item in (supporting_phrases or []) if item.strip())
    salient_terms = canonical_evidence or _salient_terms_from_documents(
        documents,
        global_document_frequency=global_document_frequency,
    )
    evaluation_terms = tuple(term.casefold().strip() for term in (*canonical_evidence, *supporting_evidence, *top_terms) if term.strip())
    unique_terms = len({term.casefold() for term in evaluation_terms})
    support_score = min(topic_size / 4.0, 1.0)
    distinctiveness_score = unique_terms / max(len(evaluation_terms), 1)
    generic_term_ratio = (
        sum(1 for term in (*top_terms, *canonical_evidence, *supporting_evidence) if term.casefold().strip() in _GENERIC_TOPIC_TOKENS)
        / max(len(top_terms) + len(canonical_evidence) + len(supporting_evidence), 1)
    )
    representative_diversity = len(set(artifact.representative_document_ids)) / max(
        min(topic_size, max(len(artifact.representative_document_ids), 1)),
        1,
    )
    distinct_support = len({document.mention_id for document in documents}) / max(topic_size, 1)
    reaction_only_burden = _metadata_ratio(documents, "reaction_only")
    low_information_burden = _metadata_ratio(documents, "low_information")
    mixed_uncertain_burden = _metadata_ratio(documents, "mixed_language_uncertain")
    chatter_burden = round(max(reaction_only_burden, low_information_burden, generic_term_ratio), 4)
    evidence_density = round(distinct_support, 4)
    recurrence_score = round(
        min(len({document.mention_id for document in documents}) / max(topic_size, 1), 1.0),
        4,
    )
    readability_score = round(
        sum(_phrase_readability_score(term) for term in canonical_evidence or top_terms) / max(len(canonical_evidence or top_terms), 1),
        4,
    )
    business_salience_score = round(
        max(
            min(
                (len(canonical_evidence or salient_terms) / max(len(top_terms), 1)) * 0.4
                + (1.0 - generic_term_ratio) * 0.35
                + readability_score * 0.25,
                1.0,
            ),
            0.0,
        ),
        4,
    )
    hashtag_only = bool(documents) and all(_has_hashtag(document.text) for document in documents)
    generic_misc = artifact.topic_key == "misc" or artifact.topic_label.casefold() == "misc"
    repeated_single_term = unique_terms <= 1 and any(len(document.normalized_text.split()) <= 3 for document in documents)
    cohesion_score = _representative_cohesion(
        documents,
        embedding_provider=embedding_provider,
        text_embedding_cache=text_embedding_cache,
    )
    distinctiveness_floor = 0.55 if topic_size >= 3 else 0.4
    cohesion_required = topic_size >= 3
    weak_topic = (
        topic_size < 2
        or unique_terms < 2
        or distinct_support < 0.45
        or (cohesion_required and cohesion_score < 0.22)
        or distinctiveness_score < distinctiveness_floor
        or readability_score < 0.48
    )
    noisy_topic = (
        hashtag_only
        or generic_misc
        or repeated_single_term
        or (topic_size < 2 and unique_terms <= 1)
        or (cohesion_required and cohesion_score < 0.14)
        or generic_term_ratio >= 0.5
        or readability_score < 0.42
    )
    score = (
        (support_score * 0.35)
        + (distinctiveness_score * 0.30)
        + (representative_diversity * 0.15)
        + (distinct_support * 0.20)
    )
    score -= generic_term_ratio * 0.18
    if hashtag_only:
        score -= 0.20
    if generic_misc:
        score -= 0.15
    if distinctiveness_score < distinctiveness_floor:
        score -= 0.12
    quality_score = round(max(min(score, 1.0), 0.0), 4)
    usefulness_score = round(max(quality_score - (0.20 if noisy_topic else 0.0) - (0.10 if weak_topic else 0.0), 0.0), 4)
    reason_flags: list[str] = []
    if weak_topic:
        reason_flags.append("low_support_or_low_term_diversity")
    if noisy_topic:
        reason_flags.append("noisy_cluster")
    if hashtag_only:
        reason_flags.append("hashtag_only_cluster")
    if generic_misc:
        reason_flags.append("generic_misc_cluster")
    if distinct_support < 0.45:
        reason_flags.append("low_distinct_support")
    if repeated_single_term:
        reason_flags.append("low_distinctiveness")
    if generic_term_ratio >= 0.4:
        reason_flags.append("generic_topic_terms")
    if readability_score < 0.52:
        reason_flags.append("low_business_readability")
    if cohesion_required and cohesion_score < 0.22:
        reason_flags.append("low_representative_cohesion")
    if reaction_only_burden >= 0.4:
        reason_flags.append("reaction_only_burden")
    if low_information_burden >= 0.45:
        reason_flags.append("low_information_burden")
    if mixed_uncertain_burden >= 0.4:
        reason_flags.append("mixed_language_uncertain_burden")
    reportability_score = round(
        max(
            min(
                quality_score * 0.34
                + usefulness_score * 0.26
                + evidence_density * 0.16
                + recurrence_score * 0.12
                + business_salience_score * 0.12
                - reaction_only_burden * 0.18
                - low_information_burden * 0.16
                - mixed_uncertain_burden * 0.08,
                1.0,
            ),
            0.0,
        ),
        4,
    )
    reporting_status = "reportable"
    if hashtag_only or reaction_only_burden >= 0.6 or low_information_burden >= 0.65:
        reporting_status = "suppressed"
    elif noisy_topic or weak_topic or reportability_score < 0.52:
        reporting_status = "discovery_only"
    return TopicQualityAssessment(
        quality_score=quality_score,
        usefulness_score=usefulness_score,
        reportability_score=reportability_score,
        reporting_status=reporting_status,
        weak_topic=weak_topic,
        noisy_topic=noisy_topic,
        hashtag_only=hashtag_only,
        chatter_burden=chatter_burden,
        reaction_only_burden=reaction_only_burden,
        low_information_burden=low_information_burden,
        evidence_density=evidence_density,
        recurrence_score=recurrence_score,
        business_salience_score=business_salience_score,
        salient_terms=tuple((canonical_evidence or salient_terms)[:5]),
        reason_flags=tuple(reason_flags),
    )


def topic_stability_score(
    current: TopicArtifactFact,
    previous: Sequence[TopicArtifactFact],
    *,
    embedding_provider: EmbeddingProvider | None = None,
    text_embedding_cache: dict[str, tuple[float, ...]] | None = None,
    lineage_level: Literal["a", "b", "c"] = "b",
) -> tuple[float, TopicArtifactFact | None]:
    best_score = 0.0
    best_match: TopicArtifactFact | None = None
    for artifact in previous:
        score = _topic_similarity(
            current,
            artifact,
            embedding_provider=embedding_provider,
            text_embedding_cache=text_embedding_cache,
            lineage_level=lineage_level,
        )
        if score > best_score:
            best_score = score
            best_match = artifact
    return round(best_score, 4), best_match


def growth_delta_for_topic(
    current: TopicArtifactFact,
    previous_match: TopicArtifactFact | None,
) -> float:
    if previous_match is None:
        return float(current.topic_size)
    return float(current.topic_size - previous_match.topic_size)


def emerging_topic_flag(current: TopicArtifactFact) -> bool:
    growth_delta = current.growth_delta or 0.0
    growth_signal = growth_delta >= 2.0 or (
        growth_delta >= 1.0
        and current.topic_size >= 4
        and (current.quality_score or 0.0) >= 0.8
        and (current.usefulness_score or 0.0) >= 0.7
    )
    return bool(
        current.growth_delta is not None
        and growth_signal
        and current.topic_size >= 2
        and (current.quality_score or 0.0) >= 0.58
        and (current.usefulness_score or 0.0) >= 0.45
        and not current.weak_topic
        and not current.noisy_topic
        and (current.stability_score or 0.0) < 0.45
    )


def select_reporting_topic_label(
    current: TopicArtifactFact,
    *,
    previous_match: TopicArtifactFact | None,
    documents: Sequence[TopicDocument],
    embedding_provider: EmbeddingProvider | None = None,
    global_document_frequency: dict[str, int] | None = None,
    text_embedding_cache: dict[str, tuple[float, ...]] | None = None,
) -> TopicLabelChoice:
    reporting_topic_key = current.effective_topic_key or current.reviewed_topic_id or current.topic_key
    if current.reviewed_label_override:
        return TopicLabelChoice(
            reporting_topic_key=reporting_topic_key,
            reporting_topic_label=current.reviewed_label_override,
            label_source="reviewed_label",
            label_confidence=0.99,
        )
    controlled_label = current.effective_topic_label or current.topic_label
    if controlled_label and controlled_label.casefold() != "misc":
        return TopicLabelChoice(
            reporting_topic_key=reporting_topic_key,
            reporting_topic_label=controlled_label,
            label_source="effective_topic_label" if current.effective_topic_label else "provider_controlled_label",
            label_confidence=0.95 if current.effective_topic_label else 0.9,
        )
    if previous_match is not None:
        previous_label = (
            previous_match.reviewed_label_override
            or previous_match.reporting_topic_label
            or previous_match.effective_topic_label
        )
        previous_key = (
            previous_match.reporting_topic_key
            or previous_match.effective_topic_key
            or previous_match.reviewed_topic_id
            or previous_match.topic_key
        )
        previous_health = assess_topic_label_health(previous_label, previous_match) if previous_label else None
        if (
            previous_label
            and previous_key == reporting_topic_key
            and previous_health is not None
            and previous_health.score >= 0.72
            and (
                previous_match.label_source in {
                    "reviewed_label",
                    "effective_topic_label",
                    "provider_controlled_label",
                    "lineage_controlled_label",
                    "fallback_topic_catalog",
                }
                or previous_match.reviewed_label_override is not None
            )
        ):
            lineage_confidence = 0.88 if (current.stability_score or 0.0) >= 0.35 else 0.8
            return TopicLabelChoice(
                reporting_topic_key=reporting_topic_key,
                reporting_topic_label=previous_label,
                label_source="lineage_controlled_label",
                label_confidence=lineage_confidence,
            )
    fallback_label = reporting_topic_key.replace("_", " ").title()
    if embedding_provider is not None and documents:
        del global_document_frequency, text_embedding_cache
        return TopicLabelChoice(
            reporting_topic_key=reporting_topic_key,
            reporting_topic_label=fallback_label,
            label_source="fallback_topic_catalog",
            label_confidence=0.72,
        )
    return TopicLabelChoice(
        reporting_topic_key=reporting_topic_key,
        reporting_topic_label=fallback_label,
        label_source="fallback_topic_catalog",
        label_confidence=0.72,
    )


def assess_topic_label_health(label: str | None, artifact: TopicArtifactFact) -> TopicLabelHealth:
    if label is None or not label.strip():
        return TopicLabelHealth(score=0.0, flags=("missing_label",))
    normalized = label.casefold().strip()
    tokens = [token for token in normalized.replace("/", " ").split() if token]
    flags: list[str] = []
    score = 0.45
    if label in {artifact.reviewed_label_override, artifact.effective_topic_label, artifact.topic_label}:
        score += 0.38
    if 1 < len(tokens) <= 6:
        score += 0.1
    if any(marker in normalized for marker in ("http", "https", "www", "example", "link")):
        score -= 0.35
        flags.append("url_fragment_label")
    generic_ratio = (
        sum(1 for token in tokens if token in _GENERIC_TOPIC_TOKENS) / max(len(tokens), 1)
    )
    if generic_ratio >= 0.4:
        score -= 0.24
        flags.append("generic_topic_label")
    if "/" in label and label not in {artifact.reviewed_label_override, artifact.effective_topic_label, artifact.topic_label}:
        score -= 0.12
        flags.append("representative_phrase_like")
    if len(tokens) == 1 and len(tokens[0]) < 4:
        score -= 0.18
        flags.append("too_short")
    return TopicLabelHealth(score=round(max(min(score, 0.99), 0.0), 4), flags=tuple(sorted(set(flags))))


def load_latest_topic_snapshot(path: Path) -> list[TopicArtifactFact]:
    if not path.exists():
        return []
    return [
        TopicArtifactFact.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def persist_topic_artifact_snapshot(output_dir: Path, artifacts: Sequence[TopicArtifactFact]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest_topic_artifacts.jsonl"
    if not artifacts:
        write_jsonl(latest_path, [])
        return latest_path
    run_id = artifacts[0].run_id or "topic-run"
    snapshot_path = output_dir / f"topic_artifacts_{run_id}.jsonl"
    payload = [artifact.model_dump(mode="json") for artifact in sorted(artifacts, key=lambda item: item.topic_key)]
    write_jsonl(snapshot_path, payload)
    write_jsonl(latest_path, payload)
    return latest_path


def _topic_similarity(
    current: TopicArtifactFact,
    previous: TopicArtifactFact,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    text_embedding_cache: dict[str, tuple[float, ...]] | None = None,
    lineage_level: Literal["a", "b", "c"] = "b",
) -> float:
    learned_features = _topic_similarity_features(
        current,
        previous,
        embedding_provider=embedding_provider,
        text_embedding_cache=text_embedding_cache,
    )
    heuristic_score = (
        (learned_features["identity_overlap"] * 0.28)
        + (learned_features["topic_family_overlap"] * 0.14)
        + (learned_features["profile_signature_overlap"] * 0.18)
        + (learned_features["lineage_continuity"] * 0.12)
        + (learned_features["entity_profile_overlap"] * 0.1)
        + (learned_features["entity_anchor_overlap"] * 0.07)
        + (learned_features["aspect_profile_overlap"] * 0.05)
        + (learned_features["issue_profile_overlap"] * 0.03)
        + (learned_features["evidence_phrase_overlap"] * 0.01)
        + (learned_features["semantic_profile_overlap"] * 0.01)
        + (learned_features["purity_continuity"] * 0.005)
        + (learned_features["issue_leak_continuity"] * 0.005)
        + (learned_features["label_overlap"] * 0.005)
        + (learned_features["reviewed_label_overlap"] * 0.01)
    )
    if lineage_level == "a":
        return round(min(heuristic_score, 1.0), 4)
    verifier_score = (
        learned_features["identity_overlap"] * 0.22
        + learned_features["topic_family_overlap"] * 0.14
        + learned_features["profile_signature_overlap"] * 0.2
        + learned_features["entity_profile_overlap"] * 0.12
        + learned_features["entity_anchor_overlap"] * 0.1
        + learned_features["aspect_profile_overlap"] * 0.06
        + learned_features["issue_profile_overlap"] * 0.04
        + learned_features["semantic_profile_overlap"] * 0.09
        + learned_features["lineage_continuity"] * 0.03
    )
    blended = (verifier_score * 0.72) + (heuristic_score * 0.28)
    if lineage_level == "c" and learned_features["identity_overlap"] < 1.0:
        blended += learned_features["semantic_profile_overlap"] * 0.08
        blended += learned_features["entity_anchor_overlap"] * 0.04
    return round(min(blended, 1.0), 4)


def _topic_similarity_features(
    current: TopicArtifactFact,
    previous: TopicArtifactFact,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    text_embedding_cache: dict[str, tuple[float, ...]] | None = None,
) -> dict[str, float]:
    profile_signature_overlap = 1.0 if (
        current.topic_profile_signature is not None
        and previous.topic_profile_signature is not None
        and current.topic_profile_signature == previous.topic_profile_signature
    ) else 0.0
    identity_overlap = 1.0 if (
        (current.effective_topic_key or current.reviewed_topic_id or current.topic_key)
        == (previous.effective_topic_key or previous.reviewed_topic_id or previous.topic_key)
    ) else 0.0
    topic_family_overlap = 1.0 if (
        (current.topic_family or current.topic_key)
        == (previous.topic_family or previous.topic_key)
    ) else 0.0
    label_overlap = 1.0 if (
        (current.reporting_topic_label or current.effective_topic_label or current.topic_label).casefold()
        == (previous.reporting_topic_label or previous.effective_topic_label or previous.topic_label).casefold()
    ) else 0.0
    reviewed_label_overlap = 1.0 if (
        current.reviewed_label_override is not None
        and previous.reviewed_label_override is not None
        and current.reviewed_label_override.casefold() == previous.reviewed_label_override.casefold()
    ) else 0.0
    lineage_continuity = 1.0 if topic_lineage_matches(current, previous) else 0.0
    entity_profile_overlap = _jaccard(
        set(current.top_canonical_entity_ids),
        set(previous.top_canonical_entity_ids),
    )
    entity_anchor_overlap = _jaccard(
        set(current.entity_anchor_phrases),
        set(previous.entity_anchor_phrases),
    )
    aspect_profile_overlap = _jaccard(set(current.aspect_profile), set(previous.aspect_profile))
    issue_profile_overlap = _jaccard(set(current.issue_profile), set(previous.issue_profile))
    evidence_phrase_overlap = _jaccard(
        set(current.canonical_evidence_phrases or current.top_terms),
        set(previous.canonical_evidence_phrases or previous.top_terms),
    )
    semantic_profile_overlap = _semantic_profile_similarity(
        current,
        previous,
        embedding_provider=embedding_provider,
        text_embedding_cache=text_embedding_cache,
    )
    purity_continuity = max(
        0.0,
        1.0 - abs((current.artifact_purity_score or 0.0) - (previous.artifact_purity_score or 0.0)),
    )
    issue_leak_continuity = max(
        0.0,
        1.0 - abs((current.issue_leak_rate or 0.0) - (previous.issue_leak_rate or 0.0)),
    )
    return {
        "identity_overlap": identity_overlap,
        "topic_family_overlap": topic_family_overlap,
        "profile_signature_overlap": profile_signature_overlap,
        "lineage_continuity": lineage_continuity,
        "entity_profile_overlap": entity_profile_overlap,
        "entity_anchor_overlap": entity_anchor_overlap,
        "aspect_profile_overlap": aspect_profile_overlap,
        "issue_profile_overlap": issue_profile_overlap,
        "evidence_phrase_overlap": evidence_phrase_overlap,
        "semantic_profile_overlap": semantic_profile_overlap,
        "purity_continuity": purity_continuity,
        "issue_leak_continuity": issue_leak_continuity,
        "label_overlap": label_overlap,
        "reviewed_label_overlap": reviewed_label_overlap,
        "current_reportability": current.reportability_score or 0.0,
        "previous_reportability": previous.reportability_score or 0.0,
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _has_hashtag(text: str) -> bool:
    return any(token.startswith("#") and len(token) > 1 for token in text.split())


def _metadata_ratio(documents: Sequence[TopicDocument], flag: str) -> float:
    if not documents:
        return 0.0
    hits = 0
    for document in documents:
        flags = document.metadata.get("text_quality_flags")
        if (isinstance(flags, list) and flag in flags) or document.metadata.get("text_quality_label") == flag:
            hits += 1
    return round(hits / len(documents), 4)


def _representative_cohesion(
    documents: Sequence[TopicDocument],
    *,
    embedding_provider: EmbeddingProvider | None = None,
    text_embedding_cache: dict[str, tuple[float, ...]] | None = None,
) -> float:
    texts = [document.normalized_text for document in documents[:4] if document.normalized_text.strip()]
    if len(texts) <= 1:
        return 1.0 if texts else 0.0
    scores: list[float] = []
    vector_lookup = _lookup_embeddings(
        texts,
        embedding_provider=embedding_provider,
        purpose=EmbeddingPurpose.CLUSTERING,
        text_embedding_cache=text_embedding_cache,
    )
    for index, left in enumerate(texts):
        for right in texts[index + 1 :]:
            lexical = _jaccard(set(left.split()), set(right.split()))
            if embedding_provider is None:
                scores.append(lexical)
                continue
            left_vector = vector_lookup.get(left)
            right_vector = vector_lookup.get(right)
            if left_vector is None or right_vector is None:
                scores.append(lexical)
                continue
            semantic = sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
            scores.append((lexical * 0.35) + (semantic * 0.65))
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def _semantic_profile_similarity(
    current: TopicArtifactFact,
    previous: TopicArtifactFact,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    text_embedding_cache: dict[str, tuple[float, ...]] | None = None,
) -> float:
    current_profile = " || ".join(
        [
            *current.entity_anchor_phrases,
            *(current.canonical_evidence_phrases or current.top_terms),
            *current.aspect_profile,
            *current.issue_profile,
        ]
    ).strip()
    previous_profile = " || ".join(
        [
            *previous.entity_anchor_phrases,
            *(previous.canonical_evidence_phrases or previous.top_terms),
            *previous.aspect_profile,
            *previous.issue_profile,
        ]
    ).strip()
    if not current_profile or not previous_profile:
        return 0.0
    if embedding_provider is None:
        return _jaccard(set(current_profile.split()), set(previous_profile.split()))
    vectors = _lookup_embeddings(
        [current_profile, previous_profile],
        embedding_provider=embedding_provider,
        purpose=EmbeddingPurpose.CLUSTERING,
        text_embedding_cache=text_embedding_cache,
    )
    current_vector = vectors.get(current_profile)
    previous_vector = vectors.get(previous_profile)
    if current_vector is None or previous_vector is None:
        return 0.0
    return round(sum(a * b for a, b in zip(current_vector, previous_vector, strict=True)), 4)


def _salient_terms_from_documents(
    documents: Sequence[TopicDocument],
    *,
    global_document_frequency: dict[str, int] | None = None,
) -> tuple[str, ...]:
    if not documents:
        return ()
    topic_doc_frequency: Counter[str] = Counter()
    topic_term_frequency: Counter[str] = Counter()
    for document in documents:
        tokens = [
            token
            for token in document.normalized_text.split()
            if _is_salient_topic_token(token)
        ]
        topic_doc_frequency.update(set(tokens))
        topic_term_frequency.update(tokens)
    if not topic_doc_frequency:
        return ()
    scored: list[tuple[str, float]] = []
    for token, doc_frequency in topic_doc_frequency.items():
        corpus_df = max((global_document_frequency or {}).get(token, 1), 1)
        term_frequency = topic_term_frequency[token]
        specificity = doc_frequency / corpus_df
        score = (specificity * 0.7) + (min(term_frequency / max(len(documents), 1), 1.0) * 0.3)
        scored.append((token, score))
    scored.sort(key=lambda item: (-item[1], -topic_term_frequency[item[0]], item[0]))
    return tuple(token for token, _ in scored[:5])


def _is_salient_topic_token(token: str) -> bool:
    cleaned = token.strip().casefold()
    if len(cleaned) < 4 or cleaned.startswith("#") or cleaned in _GENERIC_TOPIC_TOKENS:
        return False
    return not cleaned.isdigit()


def _phrase_readability_score(phrase: str) -> float:
    cleaned = phrase.casefold().strip()
    if not cleaned:
        return 0.0
    tokens = [token for token in cleaned.replace("/", " ").split() if token]
    if not tokens:
        return 0.0
    score = 0.36
    if len(tokens) >= 2:
        score += 0.18
    if any(token in _GENERIC_TOPIC_TOKENS for token in tokens):
        score -= 0.18
    if any(token.startswith("#") or token.startswith("@") for token in tokens):
        score -= 0.34
    if any("http" in token for token in tokens):
        score -= 0.34
    if max((len(token) for token in tokens), default=0) >= 5:
        score += 0.14
    if any(character.isdigit() for character in cleaned):
        score += 0.08
    return round(max(min(score, 1.0), 0.0), 4)


def _lookup_embeddings(
    texts: Sequence[str],
    *,
    embedding_provider: EmbeddingProvider | None,
    purpose: EmbeddingPurpose,
    text_embedding_cache: dict[str, tuple[float, ...]] | None = None,
) -> dict[str, tuple[float, ...]]:
    if embedding_provider is None:
        return {}
    text_embedding_cache = text_embedding_cache if text_embedding_cache is not None else {}
    missing = [text for text in dict.fromkeys(texts) if text and text not in text_embedding_cache]
    if missing:
        vectors = embedding_provider.embed_texts(missing, purpose=purpose)
        text_embedding_cache.update(zip(missing, vectors, strict=True))
    return {
        text: text_embedding_cache[text]
        for text in texts
        if text in text_embedding_cache
    }
