from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast

import numpy as np

from smap.canonicalization.alias import normalize_alias
from smap.core.settings import Settings
from smap.enrichers.anchors import (
    ASPECT_SEEDS,
    ISSUE_SEEDS,
    contains_first_person,
    extract_lexical_anchors,
)
from smap.enrichers.calibration import (
    clamp_confidence,
    issue_confidence_components,
    sentiment_confidence_components,
)
from smap.enrichers.models import (
    AspectOpinionFact,
    EntityFact,
    FactProvenance,
    IssueSeverity,
    IssueSignalFact,
    SentimentFact,
    TargetSentimentFact,
)
from smap.enrichers.segmentation import SemanticSegmenter
from smap.enrichers.semantic_assist import (
    PhraseWindow,
    SemanticAssistMatch,
    build_phrase_windows,
    collect_mapping_matches,
    mapping_support_by_label,
    rank_taxonomy_candidates_for_text,
)
from smap.enrichers.semantic_models import (
    AnchorType,
    AspectOpinionHypothesis,
    EvidenceMode,
    EvidenceScope,
    EvidenceSpan,
    IssueSignalHypothesis,
    MentionSentimentHypothesis,
    ScoreComponent,
    SemanticAnchor,
    SemanticHypothesisBatch,
    SemanticSegment,
    TargetReference,
    TargetSentimentHypothesis,
)
from smap.hil.feedback_store import PromotedSemanticKnowledgeRecord
from smap.hil.semantic_feedback import apply_promoted_semantic_feedback
from smap.normalization.models import MentionRecord
from smap.ontology.models import OntologyRegistry
from smap.ontology.prototypes import PrototypeRegistry
from smap.providers.base import (
    EmbeddingProvider,
    EmbeddingPurpose,
    RerankerProvider,
    TaxonomyMappingCandidate,
    TaxonomyMappingProvider,
)
from smap.threads.models import MentionContext

SentimentValue = Literal["positive", "negative", "neutral", "mixed"]

_ASPECT_TO_ISSUE = {
    "price": "pricing_value_concern",
    "quality": "product_defect",
    "performance": "performance_problem",
    "durability": "product_defect",
    "availability": "availability_problem",
    "customer_service": "service_problem",
    "delivery": "delivery_problem",
    "trust": "trust_concern",
    "safety": "safety_concern",
    "battery": "performance_problem",
    "charging": "performance_problem",
    "usability": "usability_problem",
}

_SCOPE_MATCH_CHARS = 48
_TARGET_ENTITY_TYPES = {
    "brand",
    "product",
    "product_line",
    "category",
    "concept",
    "retailer",
    "organization",
    "person",
    "location",
    "campaign",
    "influencer",
}
_TARGET_SPECIFICITY = {
    "product": 4,
    "product_line": 4,
    "brand": 3,
    "retailer": 3,
    "organization": 3,
    "campaign": 2,
    "influencer": 2,
    "category": 1,
    "concept": 1,
    "person": 1,
    "location": 1,
}
_HIGH_SIGNAL_CANONICAL_MATCHES = {
    "exact_alias",
    "normalized_alias",
    "compact_alias",
}
_CONTEXTUAL_BUSINESS_TARGET_TYPES = {
    "brand",
    "organization",
    "retailer",
    "location",
    "facility",
    "person",
}
_HIGH_SIGNAL_TARGET_DISCOVERY = {
    "alias_scan",
    "title_span",
    "model_code_pattern",
}
_SEMANTIC_ASSIST_MIN_SCORE = 0.58
_CORROBORATION_SIMILARITY_THRESHOLD = 0.52
_CORROBORATION_PAIRWISE_MAX_GROUP = 96
_PHRASE_WINDOW_MAX_SIZE = 4
_SEMANTIC_CACHE_LIMITS = {
    "mapping_candidate": 12000,
    "segment": 4000,
    "phrase_window": 4000,
    "semantic_assist_match": 8000,
    "taxonomy_rerank": 4000,
    "issue_mode_rerank": 2000,
    "pairwise_similarity": 2000,
    "query_embedding": 8000,
}


@dataclass(slots=True)
class SemanticInferenceResult:
    mention_sentiments: list[SentimentFact]
    target_sentiments: list[TargetSentimentFact]
    aspect_opinions: list[AspectOpinionFact]
    issue_signals: list[IssueSignalFact]


@dataclass(frozen=True, slots=True)
class PrototypeCandidateScore:
    label: str
    total_score: float
    lexical_score: float
    semantic_score: float


class SemanticInferenceEnricher:
    provider_version = "semantic-local-v4"

    def __init__(
        self,
        *,
        ontology_registry: OntologyRegistry | None = None,
        prototype_registry: PrototypeRegistry | None = None,
        taxonomy_mapping_provider: TaxonomyMappingProvider | None = None,
        reranker_provider: RerankerProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        promoted_semantic_knowledge: list[PromotedSemanticKnowledgeRecord] | None = None,
        semantic_assist_enabled: bool = True,
        semantic_hypothesis_rerank_enabled: bool = True,
        semantic_corroboration_enabled: bool = True,
        parallel_workers: int = 1,
        parallel_min_mentions: int = 64,
        parallel_chunk_size: int = 32,
        parallel_settings: Settings | None = None,
        runtime_diagnostics_enabled: bool = False,
        orchestration_cache_enabled: bool = True,
    ) -> None:
        self.segmenter = SemanticSegmenter()
        self.ontology_registry = ontology_registry
        self.taxonomy_mapping_provider = taxonomy_mapping_provider
        self.reranker_provider = reranker_provider
        self.embedding_provider = embedding_provider
        self.prototype_registry = prototype_registry or (
            PrototypeRegistry(ontology_registry, embedding_provider=embedding_provider)
            if ontology_registry is not None
            else None
        )
        self.aspect_seed_map = self.prototype_registry.aspect_seed_map() if self.prototype_registry is not None else ASPECT_SEEDS
        self.issue_seed_map = self.prototype_registry.issue_seed_map() if self.prototype_registry is not None else ISSUE_SEEDS
        self._non_target_seed_terms = {
            normalize_alias(term)
            for seed_group in [*self.aspect_seed_map.values(), *self.issue_seed_map.values()]
            for term in seed_group
        }
        self.promoted_semantic_knowledge = promoted_semantic_knowledge or []
        self.semantic_assist_enabled = semantic_assist_enabled
        self.semantic_hypothesis_rerank_enabled = semantic_hypothesis_rerank_enabled
        self.semantic_corroboration_enabled = semantic_corroboration_enabled
        self.parallel_workers = max(parallel_workers, 1)
        self.parallel_min_mentions = max(parallel_min_mentions, 1)
        self.parallel_chunk_size = max(parallel_chunk_size, 1)
        self._runtime_diagnostics_enabled = runtime_diagnostics_enabled
        self._orchestration_cache_enabled = orchestration_cache_enabled
        self._parallel_settings_payload = (
            parallel_settings.model_dump(mode="json")
            if parallel_settings is not None
            else None
        )
        self._parallel_ontology_payload = (
            ontology_registry.model_dump(mode="json")
            if ontology_registry is not None
            else None
        )
        self._mapping_candidate_cache: dict[tuple[tuple[str, ...], str, int], list[TaxonomyMappingCandidate]] = {}
        self._segment_cache: dict[str, list[SemanticSegment]] = {}
        self._phrase_window_cache: dict[tuple[str, str, int], list[tuple[str, int, int, str, tuple[str, ...], int]]] = {}
        self._semantic_assist_match_cache: dict[
            tuple[str, str, tuple[str, ...], float],
            list[tuple[str, str, str, str, int, int, float, str, str, bool, float]],
        ] = {}
        self._taxonomy_rerank_cache: dict[tuple[str, tuple[str, ...], int], dict[str, float]] = {}
        self._issue_mode_rerank_cache: dict[tuple[str, tuple[str, ...]], dict[EvidenceMode, float]] = {}
        self._pairwise_similarity_cache: dict[tuple[str, ...], list[list[float]]] = {}
        self._query_embedding_cache: dict[tuple[EmbeddingPurpose, str], tuple[float, ...]] = {}
        self._cache_stats: dict[str, dict[str, int]] = {}
        self._cache_entry_overrides: dict[str, int] = {}
        self._runtime_timings: dict[str, float] = {}
        self._runtime_counters: dict[str, int] = {}
        self._reset_runtime_stats()

    def enrich(
        self,
        mentions: list[MentionRecord],
        contexts: list[MentionContext],
        entity_facts: list[EntityFact],
    ) -> SemanticInferenceResult:
        self._clear_runtime_caches()
        if self._runtime_diagnostics_enabled:
            self._reset_runtime_stats()
            self._increment_counter("mention_count", len(mentions))
        mention_map = {mention.mention_id: mention for mention in mentions}
        explicit_targets_by_mention = self._explicit_targets_by_mention(mentions, entity_facts)
        hypotheses = self._collect_hypotheses(
            mentions=mentions,
            contexts=contexts,
            explicit_targets_by_mention=explicit_targets_by_mention,
        )
        if self._runtime_diagnostics_enabled:
            self._merge_runtime_snapshot(hypotheses.runtime_stats)
        mention_hypotheses = hypotheses.mention_sentiments
        target_hypotheses = hypotheses.target_sentiments
        aspect_hypotheses = hypotheses.aspect_opinions
        issue_hypotheses = hypotheses.issue_signals

        if self.semantic_corroboration_enabled:
            if self._runtime_diagnostics_enabled:
                started_at = perf_counter()
                issue_hypotheses = self._apply_thread_corroboration(
                    mentions=mention_map,
                    issue_hypotheses=issue_hypotheses,
                )
                self._record_timing("corroboration_issue", perf_counter() - started_at)
                started_at = perf_counter()
                aspect_hypotheses = self._apply_aspect_corroboration(
                    mentions=mention_map,
                    aspect_hypotheses=aspect_hypotheses,
                )
                self._record_timing("corroboration_aspect", perf_counter() - started_at)
            else:
                issue_hypotheses = self._apply_thread_corroboration(
                    mentions=mention_map,
                    issue_hypotheses=issue_hypotheses,
                )
                aspect_hypotheses = self._apply_aspect_corroboration(
                    mentions=mention_map,
                    aspect_hypotheses=aspect_hypotheses,
                )
        if self.semantic_hypothesis_rerank_enabled:
            aspect_hypotheses = self._rerank_aspect_hypotheses(
                mentions=mention_map,
                hypotheses=aspect_hypotheses,
            )
            issue_hypotheses = self._rerank_issue_hypotheses(
                mentions=mention_map,
                hypotheses=issue_hypotheses,
            )

        if self._runtime_diagnostics_enabled:
            started_at = perf_counter()
            result = SemanticInferenceResult(
                mention_sentiments=[self._build_mention_sentiment_fact(item, mention_map[item.mention_id]) for item in mention_hypotheses],
                target_sentiments=[self._build_target_sentiment_fact(item, mention_map[item.mention_id]) for item in target_hypotheses],
                aspect_opinions=[self._build_aspect_fact(item, mention_map[item.mention_id]) for item in aspect_hypotheses],
                issue_signals=[self._build_issue_fact(item, mention_map[item.mention_id]) for item in issue_hypotheses],
            )
            self._record_timing("final_fact_assembly", perf_counter() - started_at)
        else:
            result = SemanticInferenceResult(
                mention_sentiments=[self._build_mention_sentiment_fact(item, mention_map[item.mention_id]) for item in mention_hypotheses],
                target_sentiments=[self._build_target_sentiment_fact(item, mention_map[item.mention_id]) for item in target_hypotheses],
                aspect_opinions=[self._build_aspect_fact(item, mention_map[item.mention_id]) for item in aspect_hypotheses],
                issue_signals=[self._build_issue_fact(item, mention_map[item.mention_id]) for item in issue_hypotheses],
            )
        if self.promoted_semantic_knowledge:
            applied = apply_promoted_semantic_feedback(
                aspect_facts=result.aspect_opinions,
                issue_facts=result.issue_signals,
                promoted_records=self.promoted_semantic_knowledge,
            )
            return SemanticInferenceResult(
                mention_sentiments=result.mention_sentiments,
                target_sentiments=result.target_sentiments,
                aspect_opinions=applied.aspect_facts,
                issue_signals=applied.issue_facts,
            )
        return result

    def collect_hypotheses_for_chunk(
        self,
        *,
        mentions: list[MentionRecord],
        contexts: list[MentionContext],
        explicit_targets_by_mention: dict[str, list[TargetReference]],
    ) -> SemanticHypothesisBatch:
        if self._runtime_diagnostics_enabled:
            self._reset_runtime_stats()
            self._increment_counter("mention_count", len(mentions))
        batch = self._collect_hypotheses_serial(
            mentions=mentions,
            contexts=contexts,
            explicit_targets_by_mention=explicit_targets_by_mention,
        )
        if self._runtime_diagnostics_enabled:
            batch.runtime_stats = self.semantic_runtime_snapshot()
        return batch

    def _clear_runtime_caches(self) -> None:
        self._mapping_candidate_cache.clear()
        self._segment_cache.clear()
        self._phrase_window_cache.clear()
        self._semantic_assist_match_cache.clear()
        self._taxonomy_rerank_cache.clear()
        self._issue_mode_rerank_cache.clear()
        self._pairwise_similarity_cache.clear()
        self._query_embedding_cache.clear()

    def _reset_runtime_stats(self) -> None:
        if not self._runtime_diagnostics_enabled:
            self._cache_stats = {}
            self._cache_entry_overrides = {}
            self._runtime_timings = {}
            self._runtime_counters = {}
            return
        self._cache_stats = {
            "mapping_candidate": {"hits": 0, "misses": 0, "stores": 0},
            "segment": {"hits": 0, "misses": 0, "stores": 0},
            "phrase_window": {"hits": 0, "misses": 0, "stores": 0},
            "semantic_assist_match": {"hits": 0, "misses": 0, "stores": 0},
            "taxonomy_rerank": {"hits": 0, "misses": 0, "stores": 0},
            "issue_mode_rerank": {"hits": 0, "misses": 0, "stores": 0},
            "pairwise_similarity": {"hits": 0, "misses": 0, "stores": 0},
            "query_embedding": {"hits": 0, "misses": 0, "stores": 0},
        }
        self._cache_entry_overrides = {}
        self._runtime_timings = {
            "segmentation": 0.0,
            "phrase_window_build": 0.0,
            "semantic_assist_aspect": 0.0,
            "semantic_assist_issue": 0.0,
            "target_sentiment_path": 0.0,
            "taxonomy_rerank": 0.0,
            "issue_mode_rerank": 0.0,
            "pairwise_similarity": 0.0,
            "corroboration_issue": 0.0,
            "corroboration_aspect": 0.0,
            "semantic_parallel_merge": 0.0,
            "final_fact_assembly": 0.0,
        }
        self._runtime_counters = {
            "mention_count": 0,
            "segment_count": 0,
            "phrase_window_count": 0,
            "semantic_assist_aspect_calls": 0,
            "semantic_assist_issue_calls": 0,
            "semantic_assist_match_count": 0,
            "taxonomy_rerank_calls": 0,
            "issue_mode_rerank_calls": 0,
            "pairwise_similarity_group_count": 0,
            "pairwise_similarity_group_item_total": 0,
            "pairwise_similarity_group_max_size": 0,
            "semantic_parallel_batches": 0,
        }

    def _record_cache_hit(self, cache_name: str) -> None:
        if not self._runtime_diagnostics_enabled:
            return
        self._cache_stats.setdefault(cache_name, {"hits": 0, "misses": 0, "stores": 0})
        self._cache_stats[cache_name]["hits"] += 1

    def _record_cache_miss(self, cache_name: str) -> None:
        if not self._runtime_diagnostics_enabled:
            return
        self._cache_stats.setdefault(cache_name, {"hits": 0, "misses": 0, "stores": 0})
        self._cache_stats[cache_name]["misses"] += 1

    def _record_cache_store(self, cache_name: str) -> None:
        if not self._runtime_diagnostics_enabled:
            return
        self._cache_stats.setdefault(cache_name, {"hits": 0, "misses": 0, "stores": 0})
        self._cache_stats[cache_name]["stores"] += 1

    def _record_timing(self, timing_name: str, elapsed_seconds: float) -> None:
        if not self._runtime_diagnostics_enabled:
            return
        self._runtime_timings[timing_name] = round(self._runtime_timings.get(timing_name, 0.0) + elapsed_seconds, 6)

    def _increment_counter(self, counter_name: str, value: int = 1) -> None:
        if not self._runtime_diagnostics_enabled:
            return
        self._runtime_counters[counter_name] = self._runtime_counters.get(counter_name, 0) + value

    def _set_counter_max(self, counter_name: str, value: int) -> None:
        if not self._runtime_diagnostics_enabled:
            return
        self._runtime_counters[counter_name] = max(self._runtime_counters.get(counter_name, 0), value)

    def _bounded_cache_store(self, cache_name: str, cache: dict, key: object, value: object) -> None:
        max_entries = _SEMANTIC_CACHE_LIMITS.get(cache_name)
        if max_entries is not None and max_entries > 0:
            while len(cache) >= max_entries:
                cache.pop(next(iter(cache)))
        cache[key] = value
        self._record_cache_store(cache_name)

    def _merge_runtime_snapshot(self, runtime_stats: dict[str, object] | None) -> None:
        if not self._runtime_diagnostics_enabled or not runtime_stats:
            return
        for timing_name, value in dict(runtime_stats.get("timings", {})).items():
            if isinstance(value, (int, float)):
                self._record_timing(timing_name, float(value))
        for counter_name, value in dict(runtime_stats.get("counters", {})).items():
            if isinstance(value, int):
                if counter_name == "mention_count" or counter_name == "pairwise_similarity_group_max_size":
                    self._set_counter_max(counter_name, value)
                else:
                    self._increment_counter(counter_name, value)
        for cache_name, stats in dict(runtime_stats.get("cache", {})).items():
            if not isinstance(stats, dict):
                continue
            target = self._cache_stats.setdefault(cache_name, {"hits": 0, "misses": 0, "stores": 0})
            for field_name in ("hits", "misses", "stores"):
                value = stats.get(field_name)
                if isinstance(value, int):
                    target[field_name] += value
            entries = stats.get("entries")
            if isinstance(entries, int):
                self._cache_entry_overrides[cache_name] = self._cache_entry_overrides.get(cache_name, 0) + entries

    def cache_stats_snapshot(self) -> dict[str, dict[str, int]]:
        if not self._runtime_diagnostics_enabled:
            return {}
        local_entries = {
            "mapping_candidate": len(self._mapping_candidate_cache),
            "segment": len(self._segment_cache),
            "phrase_window": len(self._phrase_window_cache),
            "semantic_assist_match": len(self._semantic_assist_match_cache),
            "taxonomy_rerank": len(self._taxonomy_rerank_cache),
            "issue_mode_rerank": len(self._issue_mode_rerank_cache),
            "pairwise_similarity": len(self._pairwise_similarity_cache),
            "query_embedding": len(self._query_embedding_cache),
        }
        return {
            "mapping_candidate": {
                **self._cache_stats.get("mapping_candidate", {"hits": 0, "misses": 0, "stores": 0}),
                "entries": local_entries["mapping_candidate"] + self._cache_entry_overrides.get("mapping_candidate", 0),
            },
            "segment": {
                **self._cache_stats.get("segment", {"hits": 0, "misses": 0, "stores": 0}),
                "entries": local_entries["segment"] + self._cache_entry_overrides.get("segment", 0),
            },
            "phrase_window": {
                **self._cache_stats.get("phrase_window", {"hits": 0, "misses": 0, "stores": 0}),
                "entries": local_entries["phrase_window"] + self._cache_entry_overrides.get("phrase_window", 0),
            },
            "semantic_assist_match": {
                **self._cache_stats.get("semantic_assist_match", {"hits": 0, "misses": 0, "stores": 0}),
                "entries": local_entries["semantic_assist_match"] + self._cache_entry_overrides.get("semantic_assist_match", 0),
            },
            "taxonomy_rerank": {
                **self._cache_stats.get("taxonomy_rerank", {"hits": 0, "misses": 0, "stores": 0}),
                "entries": local_entries["taxonomy_rerank"] + self._cache_entry_overrides.get("taxonomy_rerank", 0),
            },
            "issue_mode_rerank": {
                **self._cache_stats.get("issue_mode_rerank", {"hits": 0, "misses": 0, "stores": 0}),
                "entries": local_entries["issue_mode_rerank"] + self._cache_entry_overrides.get("issue_mode_rerank", 0),
            },
            "pairwise_similarity": {
                **self._cache_stats.get("pairwise_similarity", {"hits": 0, "misses": 0, "stores": 0}),
                "entries": local_entries["pairwise_similarity"] + self._cache_entry_overrides.get("pairwise_similarity", 0),
            },
            "query_embedding": {
                **self._cache_stats.get("query_embedding", {"hits": 0, "misses": 0, "stores": 0}),
                "entries": local_entries["query_embedding"] + self._cache_entry_overrides.get("query_embedding", 0),
            },
        }

    def semantic_runtime_snapshot(self) -> dict[str, object]:
        if not self._runtime_diagnostics_enabled:
            return {}
        return {
            "timings": {key: round(value, 6) for key, value in self._runtime_timings.items()},
            "counters": dict(self._runtime_counters),
            "cache": self.cache_stats_snapshot(),
        }

    def _explicit_targets_by_mention(
        self,
        mentions: list[MentionRecord],
        entity_facts: list[EntityFact],
    ) -> dict[str, list[TargetReference]]:
        entities_by_mention: dict[str, list[EntityFact]] = defaultdict(list)
        for fact in entity_facts:
            entities_by_mention[fact.mention_id].append(fact)
        return {
            mention.mention_id: self._explicit_targets_for_mention(entities_by_mention.get(mention.mention_id, []))
            for mention in mentions
        }

    def _collect_hypotheses(
        self,
        *,
        mentions: list[MentionRecord],
        contexts: list[MentionContext],
        explicit_targets_by_mention: dict[str, list[TargetReference]],
    ) -> SemanticHypothesisBatch:
        if (
            self.parallel_workers > 1
            and len(mentions) >= self.parallel_min_mentions
            and self._parallel_settings_payload is not None
            and self._parallel_ontology_payload is not None
            and self._process_parallelism_available()
        ):
            from smap.enrichers.semantic_parallel import parallel_collect_hypotheses

            return parallel_collect_hypotheses(
                settings_payload=self._parallel_settings_payload,
                ontology_payload=self._parallel_ontology_payload,
                mentions=mentions,
                contexts=contexts,
                explicit_targets_by_mention=explicit_targets_by_mention,
                workers=self.parallel_workers,
                chunk_size=self.parallel_chunk_size,
            )
        return self._collect_hypotheses_serial(
            mentions=mentions,
            contexts=contexts,
            explicit_targets_by_mention=explicit_targets_by_mention,
        )

    def _process_parallelism_available(self) -> bool:
        main_module = sys.modules.get("__main__")
        main_file = getattr(main_module, "__file__", None)
        return bool(main_file)

    def _collect_hypotheses_serial(
        self,
        *,
        mentions: list[MentionRecord],
        contexts: list[MentionContext],
        explicit_targets_by_mention: dict[str, list[TargetReference]],
    ) -> SemanticHypothesisBatch:
        context_map = {context.mention_id: context for context in contexts}
        batch = SemanticHypothesisBatch()
        for mention in mentions:
            context = context_map.get(mention.mention_id)
            explicit_targets = explicit_targets_by_mention.get(mention.mention_id, [])
            inherited_targets = self._inherited_targets_for_mention(
                mention,
                context,
                explicit_targets_by_mention,
            )
            segment_hypotheses = self._analyze_mention(
                mention,
                explicit_targets=explicit_targets,
                inherited_targets=inherited_targets,
            )
            batch.mention_sentiments.extend(segment_hypotheses[0])
            batch.target_sentiments.extend(segment_hypotheses[1])
            batch.aspect_opinions.extend(segment_hypotheses[2])
            batch.issue_signals.extend(segment_hypotheses[3])
        return batch

    def _analyze_mention(
        self,
        mention: MentionRecord,
        *,
        explicit_targets: list[TargetReference],
        inherited_targets: list[TargetReference],
    ) -> tuple[
        list[MentionSentimentHypothesis],
        list[TargetSentimentHypothesis],
        list[AspectOpinionHypothesis],
        list[IssueSignalHypothesis],
    ]:
        segments = self._segment_cache.get(mention.raw_text)
        if segments is None:
            self._record_cache_miss("segment")
            started_at = perf_counter() if self._runtime_diagnostics_enabled else None
            segments = self.segmenter.segment(mention.raw_text)
            if started_at is not None:
                self._record_timing("segmentation", perf_counter() - started_at)
            self._bounded_cache_store("segment", self._segment_cache, mention.raw_text, segments)
        else:
            self._record_cache_hit("segment")
        if not segments:
            return ([], [], [], [])
        self._increment_counter("segment_count", len(segments))

        target_segment_hypotheses: list[TargetSentimentHypothesis] = []
        aspect_segment_hypotheses: list[AspectOpinionHypothesis] = []
        issue_segment_hypotheses: list[IssueSignalHypothesis] = []
        segment_sentiments: list[tuple[SemanticSegment, float, list[SemanticAnchor], list[str], bool]] = []

        for segment in segments:
            routing_mode = self._semantic_routing_mode(mention)
            anchors = extract_lexical_anchors(
                segment,
                aspect_seeds=self.aspect_seed_map,
                issue_seeds=self.issue_seed_map,
            )
            anchors.extend(
                self._target_anchors_for_segment(
                    mention=mention,
                    segment=segment,
                    targets=explicit_targets,
                )
            )
            if self._segment_is_low_signal(segment, anchors):
                aspect_supports: dict[str, list[SemanticAssistMatch]] = {}
                issue_supports: dict[str, list[SemanticAssistMatch]] = {}
                generated_anchors: list[SemanticAnchor] = []
            else:
                aspect_supports, issue_supports, generated_anchors = self._semantic_assist_supports(segment, anchors)
            anchors.extend(generated_anchors)
            anchors.sort(key=lambda item: (item.start, item.end, item.anchor_type.value))
            polarity_score, polarity_supports, negated = self._segment_polarity(anchors)
            uncertainty_flags = self._uncertainty_flags(segment, anchors)
            segment_sentiments.append((segment, polarity_score, anchors, uncertainty_flags, negated))

            target_segment_hypotheses.extend(
                self._timed_segment_target_sentiments(
                    mention=mention,
                    segment=segment,
                    anchors=anchors,
                    explicit_targets=explicit_targets,
                    inherited_targets=inherited_targets,
                    polarity_score=polarity_score,
                    polarity_supports=polarity_supports,
                    negated=negated,
                    uncertainty_flags=uncertainty_flags,
                    routing_mode=routing_mode,
                )
            )
            if routing_mode == "semantic_full":
                aspect_segment_hypotheses.extend(
                    self._segment_aspect_opinions(
                        mention=mention,
                        segment=segment,
                        anchors=anchors,
                        explicit_targets=explicit_targets,
                        inherited_targets=inherited_targets,
                        uncertainty_flags=uncertainty_flags,
                        semantic_supports=aspect_supports,
                        routing_mode=routing_mode,
                    )
                )
                issue_segment_hypotheses.extend(
                    self._segment_issue_signals(
                        segment=segment,
                        anchors=anchors,
                        explicit_targets=explicit_targets,
                        inherited_targets=inherited_targets,
                        uncertainty_flags=uncertainty_flags,
                        mention=mention,
                        semantic_supports=issue_supports,
                        routing_mode=routing_mode,
                    )
                )

        mention_hypothesis = self._aggregate_mention_sentiment(mention, segment_sentiments)
        target_hypotheses = self._merge_target_hypotheses(target_segment_hypotheses)
        aspect_hypotheses = self._merge_aspect_hypotheses(aspect_segment_hypotheses)
        return (
            [mention_hypothesis] if mention_hypothesis is not None else [],
            target_hypotheses,
            aspect_hypotheses,
            issue_segment_hypotheses,
        )

    def _timed_segment_target_sentiments(
        self,
        *,
        mention: MentionRecord,
        segment: SemanticSegment,
        anchors: list[SemanticAnchor],
        explicit_targets: list[TargetReference],
        inherited_targets: list[TargetReference],
        polarity_score: float,
        polarity_supports: list[SemanticAnchor],
        negated: bool,
        uncertainty_flags: list[str],
        routing_mode: str,
    ) -> list[TargetSentimentHypothesis]:
        if not self._runtime_diagnostics_enabled:
            return self._segment_target_sentiments(
                mention=mention,
                segment=segment,
                anchors=anchors,
                explicit_targets=explicit_targets,
                inherited_targets=inherited_targets,
                polarity_score=polarity_score,
                polarity_supports=polarity_supports,
                negated=negated,
                uncertainty_flags=uncertainty_flags,
                routing_mode=routing_mode,
            )
        started_at = perf_counter()
        result = self._segment_target_sentiments(
            mention=mention,
            segment=segment,
            anchors=anchors,
            explicit_targets=explicit_targets,
            inherited_targets=inherited_targets,
            polarity_score=polarity_score,
            polarity_supports=polarity_supports,
            negated=negated,
            uncertainty_flags=uncertainty_flags,
            routing_mode=routing_mode,
        )
        self._record_timing("target_sentiment_path", perf_counter() - started_at)
        return result

    def _explicit_targets_for_mention(self, entity_facts: list[EntityFact]) -> list[TargetReference]:
        canonical_targets: dict[str, TargetReference] = {}
        concept_targets: dict[str, TargetReference] = {}
        for fact in entity_facts:
            if not self._fact_has_local_target_evidence(fact):
                continue
            normalized_candidate = normalize_alias(fact.candidate_text)
            if normalized_candidate in self._non_target_seed_terms:
                continue
            if fact.canonical_entity_id is None and fact.concept_entity_id is None:
                continue
            if fact.entity_type not in _TARGET_ENTITY_TYPES:
                continue
            if fact.concept_entity_id is not None and not self._concept_fact_is_target_eligible(fact):
                continue
            if fact.canonical_entity_id is not None and not self._canonical_fact_is_target_eligible(fact):
                continue
            target_key = (
                fact.canonical_entity_id
                or fact.concept_entity_id
            )
            if not target_key:
                continue
            target = TargetReference(
                target_key=target_key,
                target_text=fact.candidate_text,
                canonical_entity_id=fact.canonical_entity_id,
                concept_entity_id=fact.concept_entity_id,
                target_kind=(
                    "canonical_entity"
                    if fact.canonical_entity_id is not None
                    else "concept"
                    if fact.concept_entity_id is not None
                    else "surface"
                ),
                entity_type=fact.entity_type,
                grounding_confidence=self._target_grounding_confidence_from_fact(fact),
            )
            if fact.canonical_entity_id is not None:
                canonical_targets.setdefault(target_key, target)
                continue
            concept_targets.setdefault(
                target_key,
                target,
            )
        return list(canonical_targets.values()) if canonical_targets else list(concept_targets.values())

    def _canonical_fact_is_target_eligible(self, fact: EntityFact) -> bool:
        if not fact.target_eligible or fact.canonical_entity_id is None:
            return False
        if fact.confidence >= 0.78:
            return True
        if fact.matched_by in _HIGH_SIGNAL_CANONICAL_MATCHES and fact.confidence >= 0.72:
            return True
        if (
            fact.matched_by == "contextual_alias"
            and fact.entity_type in _CONTEXTUAL_BUSINESS_TARGET_TYPES
            and fact.confidence >= 0.74
            and fact.knowledge_layer in {"domain", "domain_overlay", "review_overlay"}
            and any(method in _HIGH_SIGNAL_TARGET_DISCOVERY for method in fact.discovered_by)
        ):
            return True
        return fact.surface_specificity >= 0.2 and fact.confidence >= 0.68

    def _concept_fact_is_target_eligible(self, fact: EntityFact) -> bool:
        if not fact.target_eligible or fact.concept_entity_id is None:
            return False
        if self.prototype_registry is None:
            return fact.confidence >= 0.88 and fact.surface_specificity >= 0.34
        if fact.concept_entity_id not in self.prototype_registry.eligible_concept_keys():
            return False
        if fact.knowledge_layer == "base" and fact.matched_by not in _HIGH_SIGNAL_CANONICAL_MATCHES:
            return False
        if fact.matched_by == "contextual_alias":
            return fact.knowledge_layer in {"domain", "domain_overlay", "review_overlay"} and fact.confidence >= 0.9 and fact.surface_specificity >= 0.38
        return fact.confidence >= 0.88 and fact.surface_specificity >= 0.34

    def _fact_has_local_target_evidence(self, fact: EntityFact) -> bool:
        if not fact.discovered_by:
            return True
        if fact.concept_entity_id is not None and not any(
            method in {"alias_scan", "brand_context", "product_code", "phobert_ner"}
            for method in fact.discovered_by
        ):
            return False
        return any(method != "context_span" for method in fact.discovered_by)

    def _semantic_routing_mode(self, mention: MentionRecord) -> str:
        if mention.semantic_route_hint == "semantic_lite":
            return "semantic_lite"
        if mention.text_quality_label in {"reaction_only", "low_information", "spam_like", "duplicate_like"}:
            return "semantic_lite"
        return "semantic_full"

    def _target_grounding_confidence_from_fact(self, fact: EntityFact) -> float:
        confidence = 0.38
        if fact.unresolved_cluster_id is not None:
            confidence = 0.55 if fact.unresolved_cluster_size >= 2 else 0.48
        if fact.concept_entity_id is not None:
            confidence = 0.74 if self._concept_fact_is_target_eligible(fact) else 0.58
        if fact.canonical_entity_id is not None:
            confidence = max(0.84, min(fact.confidence, 0.96)) if self._canonical_fact_is_target_eligible(fact) else 0.62
        return round(confidence, 3)

    def _inherited_targets_for_mention(
        self,
        mention: MentionRecord,
        context: MentionContext | None,
        explicit_targets_by_mention: dict[str, list[TargetReference]],
    ) -> list[TargetReference]:
        if context is None:
            return []
        candidate_mentions: list[str] = []
        if mention.parent_id is not None:
            candidate_mentions.append(mention.parent_id)
        candidate_mentions.extend(reversed(context.lineage_ids))
        candidate_mentions.append(context.root_id)
        seen: dict[str, TargetReference] = {}
        inherited_from: dict[str, str] = {}
        for ancestor_id in candidate_mentions:
            if ancestor_id == mention.mention_id:
                continue
            for target in explicit_targets_by_mention.get(ancestor_id, []):
                seen.setdefault(target.target_key, target)
                inherited_from.setdefault(target.target_key, ancestor_id)
        if not seen:
            return []
        preferred_target = self._preferred_explicit_target(list(seen.values()))
        if preferred_target is None:
            return []
        return [
            TargetReference(
                target_key=preferred_target.target_key,
                target_text=preferred_target.target_text,
                canonical_entity_id=preferred_target.canonical_entity_id,
                concept_entity_id=preferred_target.concept_entity_id,
                unresolved_cluster_id=preferred_target.unresolved_cluster_id,
                target_kind=preferred_target.target_kind,
                entity_type=preferred_target.entity_type,
                grounding_confidence=preferred_target.grounding_confidence,
                inherited=True,
                inherited_from_mention_id=inherited_from[preferred_target.target_key],
            )
        ]

    def _preferred_explicit_target(
        self,
        targets: Sequence[TargetReference],
    ) -> TargetReference | None:
        ranked_targets = sorted(
            targets,
            key=lambda target: (
                _TARGET_SPECIFICITY.get(target.entity_type or "", 0),
                len(target.target_text),
                bool(target.canonical_entity_id),
            ),
            reverse=True,
        )
        if not ranked_targets:
            return None
        top_priority = _TARGET_SPECIFICITY.get(ranked_targets[0].entity_type or "", 0)
        preferred_targets = [
            target
            for target in ranked_targets
            if _TARGET_SPECIFICITY.get(target.entity_type or "", 0) == top_priority
        ]
        if len(preferred_targets) != 1:
            return None
        return preferred_targets[0]

    def _target_anchors_for_segment(
        self,
        *,
        mention: MentionRecord,
        segment: SemanticSegment,
        targets: list[TargetReference],
    ) -> list[SemanticAnchor]:
        from smap.enrichers.anchors import build_target_anchors

        anchors: list[SemanticAnchor] = []
        for target in targets:
            anchors.extend(
                build_target_anchors(
                    mention_text=mention.raw_text,
                    segment=segment,
                    target_text=target.target_text,
                    canonical_entity_id=target.canonical_entity_id,
                    concept_entity_id=target.concept_entity_id,
                    unresolved_cluster_id=target.unresolved_cluster_id,
                    entity_type=target.entity_type,
                    source="entity_fact",
                    inherited=target.inherited,
                    inherited_from_mention_id=target.inherited_from_mention_id,
                )
            )
        return anchors

    def _semantic_assist_supports(
        self,
        segment: SemanticSegment,
        anchors: list[SemanticAnchor],
    ) -> tuple[
        dict[str, list[SemanticAssistMatch]],
        dict[str, list[SemanticAssistMatch]],
        list[SemanticAnchor],
    ]:
        if not self.semantic_assist_enabled:
            return {}, {}, []
        phrase_windows = self._phrase_windows_for_segment(segment)
        if self._runtime_diagnostics_enabled:
            self._increment_counter("semantic_assist_aspect_calls")
            started_at = perf_counter()
            aspect_matches = self._semantic_assist_matches(
                segment,
                taxonomy_labels=sorted(self.aspect_seed_map),
                phrase_windows=phrase_windows,
            )
            self._record_timing("semantic_assist_aspect", perf_counter() - started_at)
            self._increment_counter("semantic_assist_issue_calls")
            started_at = perf_counter()
            issue_matches = self._semantic_assist_matches(
                segment,
                taxonomy_labels=sorted(self.issue_seed_map),
                phrase_windows=phrase_windows,
            )
            self._record_timing("semantic_assist_issue", perf_counter() - started_at)
        else:
            aspect_matches = self._semantic_assist_matches(
                segment,
                taxonomy_labels=sorted(self.aspect_seed_map),
                phrase_windows=phrase_windows,
            )
            issue_matches = self._semantic_assist_matches(
                segment,
                taxonomy_labels=sorted(self.issue_seed_map),
                phrase_windows=phrase_windows,
            )
        aspect_supports = mapping_support_by_label(aspect_matches)
        issue_supports = mapping_support_by_label(issue_matches)
        generated: list[SemanticAnchor] = []
        existing_aspects = {anchor.label for anchor in anchors if anchor.anchor_type == AnchorType.ASPECT}
        existing_issues = {anchor.label for anchor in anchors if anchor.anchor_type == AnchorType.ISSUE}
        for match in aspect_matches:
            if (
                match.taxonomy_label in existing_aspects
                or match.score < _SEMANTIC_ASSIST_MIN_SCORE
                or self._assist_match_is_ambiguous(match, aspect_matches)
            ):
                continue
            generated.append(self._assist_match_anchor(match, anchor_type=AnchorType.ASPECT, segment_id=segment.segment_id))
            existing_aspects.add(match.taxonomy_label)
        for match in issue_matches:
            if (
                match.taxonomy_label in existing_issues
                or match.score < _SEMANTIC_ASSIST_MIN_SCORE
                or self._assist_match_is_ambiguous(match, issue_matches)
            ):
                continue
            generated.append(self._assist_match_anchor(match, anchor_type=AnchorType.ISSUE, segment_id=segment.segment_id))
            existing_issues.add(match.taxonomy_label)
        return aspect_supports, issue_supports, generated

    def _assist_match_is_ambiguous(
        self,
        match: SemanticAssistMatch,
        matches: list[SemanticAssistMatch],
    ) -> bool:
        competing = [
            item
            for item in matches
            if item.start == match.start and item.end == match.end and item.taxonomy_label != match.taxonomy_label
        ]
        if not competing:
            return False
        strongest_competing = max(item.score for item in competing)
        return strongest_competing >= (match.score - 0.05)

    def _phrase_windows_for_segment(
        self,
        segment: SemanticSegment,
        *,
        max_window_size: int = _PHRASE_WINDOW_MAX_SIZE,
    ) -> list[PhraseWindow]:
        if not self._orchestration_cache_enabled:
            started_at = perf_counter() if self._runtime_diagnostics_enabled else None
            windows = build_phrase_windows(segment, max_window_size=max_window_size)
            if started_at is not None:
                self._record_timing("phrase_window_build", perf_counter() - started_at)
            self._increment_counter("phrase_window_count", len(windows))
            return windows
        cache_key = (segment.normalized_text, segment.text, max_window_size)
        cached = self._phrase_window_cache.get(cache_key)
        if cached is None:
            self._record_cache_miss("phrase_window")
            started_at = perf_counter() if self._runtime_diagnostics_enabled else None
            windows = build_phrase_windows(segment, max_window_size=max_window_size)
            if started_at is not None:
                self._record_timing("phrase_window_build", perf_counter() - started_at)
            cached = [
                (
                    window.text,
                    window.start - segment.start,
                    window.end - segment.start,
                    window.normalized_text,
                    window.variants,
                    window.token_count,
                )
                for window in windows
            ]
            self._bounded_cache_store("phrase_window", self._phrase_window_cache, cache_key, cached)
        else:
            self._record_cache_hit("phrase_window")
        windows = [
            PhraseWindow(
                text=text,
                start=segment.start + relative_start,
                end=segment.start + relative_end,
                normalized_text=normalized_text,
                variants=variants,
                token_count=token_count,
            )
            for text, relative_start, relative_end, normalized_text, variants, token_count in cached
        ]
        self._increment_counter("phrase_window_count", len(windows))
        return windows

    def _semantic_assist_matches(
        self,
        segment: SemanticSegment,
        *,
        taxonomy_labels: list[str],
        phrase_windows: list[PhraseWindow] | None = None,
    ) -> list[SemanticAssistMatch]:
        phrase_windows = phrase_windows or self._phrase_windows_for_segment(segment)
        if not self._orchestration_cache_enabled:
            matches = collect_mapping_matches(
                segment=segment,
                taxonomy_labels=taxonomy_labels,
                taxonomy_mapping_provider=self.taxonomy_mapping_provider,
                reranker_provider=self.reranker_provider,
                min_score=0.42,
                phrase_windows=phrase_windows,
                mapping_candidate_cache=self._mapping_candidate_cache,
            )
            self._increment_counter("semantic_assist_match_count", len(matches))
            return matches
        cache_key = (
            segment.normalized_text,
            segment.text,
            tuple(sorted(taxonomy_labels)),
            0.42,
        )
        cached = self._semantic_assist_match_cache.get(cache_key)
        if cached is None:
            self._record_cache_miss("semantic_assist_match")
            matches = collect_mapping_matches(
                segment=segment,
                taxonomy_labels=taxonomy_labels,
                taxonomy_mapping_provider=self.taxonomy_mapping_provider,
                reranker_provider=self.reranker_provider,
                min_score=0.42,
                phrase_windows=phrase_windows,
                mapping_candidate_cache=self._mapping_candidate_cache,
            )
            cached = [
                (
                    match.taxonomy_label,
                    match.source_text,
                    match.normalized_text,
                    match.matched_variant,
                    match.start - segment.start,
                    match.end - segment.start,
                    match.score,
                    match.provider_name,
                    match.provider_model_id,
                    match.reranked,
                    match.ambiguity_gap,
                )
                for match in matches
            ]
            self._bounded_cache_store("semantic_assist_match", self._semantic_assist_match_cache, cache_key, cached)
        else:
            self._record_cache_hit("semantic_assist_match")
        matches = [
            SemanticAssistMatch(
                taxonomy_label=taxonomy_label,
                source_text=source_text,
                normalized_text=normalized_text,
                matched_variant=matched_variant,
                start=segment.start + relative_start,
                end=segment.start + relative_end,
                score=score,
                provider_name=provider_name,
                provider_model_id=provider_model_id,
                reranked=reranked,
                ambiguity_gap=ambiguity_gap,
            )
            for (
                taxonomy_label,
                source_text,
                normalized_text,
                matched_variant,
                relative_start,
                relative_end,
                score,
                provider_name,
                provider_model_id,
                reranked,
                ambiguity_gap,
            ) in cached
        ]
        self._increment_counter("semantic_assist_match_count", len(matches))
        return matches

    def _segment_is_low_signal(
        self,
        segment: SemanticSegment,
        anchors: list[SemanticAnchor],
    ) -> bool:
        if anchors:
            return False
        normalized = normalize_alias(segment.text)
        tokens = [token for token in normalized.split() if token]
        if len(tokens) >= 3:
            return False
        return len(normalized) < 18

    def _taxonomy_rerank_scores(
        self,
        query_text: str,
        *,
        taxonomy_labels: tuple[str, ...],
        top_k: int,
    ) -> dict[str, float]:
        if self.taxonomy_mapping_provider is None or not taxonomy_labels:
            return {}
        cache_key = (query_text, taxonomy_labels, top_k)
        cached = self._taxonomy_rerank_cache.get(cache_key)
        if cached is not None:
            self._record_cache_hit("taxonomy_rerank")
            return cached
        self._record_cache_miss("taxonomy_rerank")
        self._increment_counter("taxonomy_rerank_calls")
        started_at = perf_counter() if self._runtime_diagnostics_enabled else None
        ranked = rank_taxonomy_candidates_for_text(
            text=query_text,
            taxonomy_labels=list(taxonomy_labels),
            taxonomy_mapping_provider=self.taxonomy_mapping_provider,
            reranker_provider=self.reranker_provider,
            top_k=top_k,
            mapping_candidate_cache=self._mapping_candidate_cache,
        )
        if started_at is not None:
            self._record_timing("taxonomy_rerank", perf_counter() - started_at)
        result = {candidate.taxonomy_label: candidate.score for _, candidate, _ in ranked}
        self._bounded_cache_store("taxonomy_rerank", self._taxonomy_rerank_cache, cache_key, result)
        return result

    def _assist_match_anchor(
        self,
        match: SemanticAssistMatch,
        *,
        anchor_type: AnchorType,
        segment_id: str,
    ) -> SemanticAnchor:
        return SemanticAnchor(
            anchor_type=anchor_type,
            label=match.taxonomy_label,
            normalized_text=match.normalized_text or normalize_alias(match.source_text) or match.source_text.casefold(),
            start=match.start,
            end=match.end,
            text=match.source_text,
            source="semantic_assist",
            confidence=round(min(0.84, 0.32 + (match.score * 0.5)), 3),
            segment_id=segment_id,
            metadata={
                "mapping_score": match.score,
                "provider_name": match.provider_name,
                "provider_model_id": match.provider_model_id,
                "matched_variant": match.matched_variant,
                "reranked": match.reranked,
            },
        )

    def _segment_polarity(
        self,
        anchors: list[SemanticAnchor],
    ) -> tuple[float, list[SemanticAnchor], bool]:
        polarity_anchors = [anchor for anchor in anchors if anchor.anchor_type == AnchorType.POLARITY]
        adjusted_total = 0.0
        supports: list[SemanticAnchor] = []
        negated = False
        for anchor in polarity_anchors:
            if anchor.polarity is None:
                continue
            score = self._cue_score(anchor, anchors)
            if score != anchor.polarity:
                negated = True
            adjusted_total += score
            supports.append(anchor)
        return adjusted_total, supports, negated

    def _uncertainty_flags(self, segment: SemanticSegment, anchors: list[SemanticAnchor]) -> list[str]:
        flags: list[str] = []
        if segment.question_like or any(anchor.anchor_type == AnchorType.UNCERTAINTY for anchor in anchors):
            flags.append("question_or_uncertainty")
        if any(anchor.anchor_type == AnchorType.HEARSAY for anchor in anchors):
            flags.append("hearsay_or_rumor")
        if any(anchor.anchor_type == AnchorType.COMPARISON for anchor in anchors):
            flags.append("comparison")
        if any(anchor.anchor_type == AnchorType.CONTRAST for anchor in anchors):
            flags.append("contrast")
        return flags

    def _segment_target_sentiments(
        self,
        *,
        mention: MentionRecord,
        segment: SemanticSegment,
        anchors: list[SemanticAnchor],
        explicit_targets: list[TargetReference],
        inherited_targets: list[TargetReference],
        polarity_score: float,
        polarity_supports: list[SemanticAnchor],
        negated: bool,
        uncertainty_flags: list[str],
        routing_mode: str,
    ) -> list[TargetSentimentHypothesis]:
        if not polarity_supports:
            return []

        local_target_refs = self._segment_targets(anchors, explicit_targets)
        hypotheses: list[TargetSentimentHypothesis] = []
        preferred_explicit_target = self._preferred_explicit_target(explicit_targets)
        if not local_target_refs and preferred_explicit_target is not None:
            local_target_refs = [(preferred_explicit_target, EvidenceScope.AMBIGUOUS, False)]
        elif not local_target_refs and len(inherited_targets) == 1:
            local_target_refs = [(inherited_targets[0], EvidenceScope.INHERITED, False)]

        for target, evidence_scope, has_explicit_anchor in local_target_refs:
            local_score, supporting_cues = self._score_cues_for_target(
                target=target,
                anchors=anchors,
                explicit_anchor=has_explicit_anchor,
            )
            if not supporting_cues and evidence_scope == EvidenceScope.INHERITED:
                local_score = polarity_score * 0.75
                supporting_cues = polarity_supports
            if not supporting_cues:
                continue
            sentiment = self._sentiment_label(local_score)
            components = sentiment_confidence_components(
                explicit_target=has_explicit_anchor,
                inherited_target=target.inherited,
                support_count=len(supporting_cues),
                mixed_evidence=sentiment == "mixed",
                contrastive=segment.contrastive,
                negated=negated,
                uncertain=bool(uncertainty_flags),
            )
            confidence = clamp_confidence(0.48, components)
            hypotheses.append(
                TargetSentimentHypothesis(
                    mention_id=mention.mention_id,
                    target=target,
                    sentiment=sentiment,
                    confidence=confidence,
                    score=round(abs(local_score), 3),
                    semantic_routing=routing_mode,
                    target_grounding_confidence=target.grounding_confidence,
                    corroboration_confidence=0.55 if target.inherited else 0.7,
                    evidence_scope=evidence_scope,
                    evidence_spans=[anchor.to_evidence_span() for anchor in supporting_cues],
                    score_components=components,
                    uncertainty_flags=uncertainty_flags,
                    segment_ids=[segment.segment_id],
                )
            )
        return hypotheses

    def _segment_targets(
        self,
        anchors: list[SemanticAnchor],
        explicit_targets: list[TargetReference],
    ) -> list[tuple[TargetReference, EvidenceScope, bool]]:
        target_anchors = [anchor for anchor in anchors if anchor.anchor_type == AnchorType.TARGET]
        if not target_anchors:
            return []
        refs: list[tuple[TargetReference, EvidenceScope, bool]] = []
        for anchor in target_anchors:
            target_ref = next(
                (
                    target
                    for target in explicit_targets
                    if target.target_key == str(
                        anchor.metadata.get("canonical_entity_id")
                        or anchor.metadata.get("concept_entity_id")
                        or anchor.metadata.get("unresolved_cluster_id")
                        or anchor.label
                    )
                    or normalize_alias(target.target_text)
                    == normalize_alias(str(anchor.metadata.get("target_text", "")))
                ),
                None,
            )
            if target_ref is None:
                target_ref = TargetReference(
                    target_key=str(
                        anchor.metadata.get("canonical_entity_id")
                        or anchor.metadata.get("concept_entity_id")
                        or anchor.metadata.get("unresolved_cluster_id")
                        or anchor.label
                    ),
                    target_text=str(anchor.metadata.get("target_text") or anchor.text),
                    canonical_entity_id=str(anchor.metadata.get("canonical_entity_id") or "") or None,
                    concept_entity_id=str(anchor.metadata.get("concept_entity_id") or "") or None,
                    unresolved_cluster_id=str(anchor.metadata.get("unresolved_cluster_id") or "") or None,
                    target_kind=(
                        "canonical_entity"
                        if anchor.metadata.get("canonical_entity_id")
                        else "concept"
                        if anchor.metadata.get("concept_entity_id")
                        else "unresolved_cluster"
                        if anchor.metadata.get("unresolved_cluster_id")
                        else "surface"
                    ),
                    entity_type=str(anchor.metadata.get("entity_type") or "") or None,
                    grounding_confidence=0.58,
                )
            refs.append((target_ref, EvidenceScope.LOCAL, True))
        return refs

    def _score_cues_for_target(
        self,
        *,
        target: TargetReference,
        anchors: list[SemanticAnchor],
        explicit_anchor: bool,
    ) -> tuple[float, list[SemanticAnchor]]:
        polarity_anchors = [anchor for anchor in anchors if anchor.anchor_type == AnchorType.POLARITY]
        target_anchors = [anchor for anchor in anchors if anchor.anchor_type == AnchorType.TARGET]
        if not explicit_anchor or not target_anchors:
            score = sum(self._cue_score(anchor, anchors) for anchor in polarity_anchors)
            return score, polarity_anchors
        target_anchor = next(
            (
                anchor
                for anchor in target_anchors
                if normalize_alias(str(anchor.metadata.get("target_text", anchor.text)))
                == normalize_alias(target.target_text)
            ),
            target_anchors[0],
        )
        weighted: list[tuple[SemanticAnchor, float]] = []
        for cue in polarity_anchors:
            distance = min(abs(cue.start - target_anchor.start), abs(cue.end - target_anchor.end))
            if distance > _SCOPE_MATCH_CHARS:
                continue
            weight = max(0.35, 1.0 - (distance / _SCOPE_MATCH_CHARS))
            weighted.append((cue, self._cue_score(cue, anchors) * weight))
        if not weighted:
            return 0.0, []
        return sum(score for _, score in weighted), [anchor for anchor, _ in weighted]

    def _grounded_targets_for_segment(
        self,
        *,
        anchors: list[SemanticAnchor],
        explicit_targets: list[TargetReference],
        inherited_targets: list[TargetReference],
    ) -> list[tuple[TargetReference, EvidenceScope, SemanticAnchor | None]]:
        local = [
            (target_ref, scope, next((anchor for anchor in anchors if anchor.anchor_type == AnchorType.TARGET and normalize_alias(str(anchor.metadata.get("target_text", anchor.text))) == normalize_alias(target_ref.target_text)), None))
            for target_ref, scope, _ in self._segment_targets(anchors, explicit_targets)
            if target_ref.target_kind in {"canonical_entity", "concept"}
        ]
        if local:
            return local
        preferred_explicit_target = self._preferred_explicit_target(explicit_targets)
        if preferred_explicit_target is not None:
            return [(preferred_explicit_target, EvidenceScope.AMBIGUOUS, None)]
        if len(inherited_targets) == 1 and inherited_targets[0].target_kind in {"canonical_entity", "concept"}:
            return [(inherited_targets[0], EvidenceScope.INHERITED, None)]
        return []

    def _target_window_text(
        self,
        segment: SemanticSegment,
        target_anchor: SemanticAnchor | None,
    ) -> str:
        if target_anchor is None:
            return segment.normalized_text
        relative_start = max(target_anchor.start - segment.start - 42, 0)
        relative_end = min(target_anchor.end - segment.start + 64, len(segment.text))
        if relative_end <= relative_start:
            return segment.normalized_text
        return normalize_alias(segment.text[relative_start:relative_end]) or segment.normalized_text

    def _prototype_similarity(
        self,
        query_text: str,
        prototype_text: str,
        prototype_vector: tuple[float, ...] | None = None,
    ) -> float:
        if self.embedding_provider is None:
            return 0.0
        query_vector = self._query_embedding(query_text, purpose=EmbeddingPurpose.LINKING)
        if not query_vector:
            return 0.0
        target_vector = prototype_vector
        if target_vector is None:
            target_vector = self._query_embedding(prototype_text, purpose=EmbeddingPurpose.LINKING)
        if not target_vector:
            return 0.0
        return max(sum(a * b for a, b in zip(query_vector, target_vector, strict=True)), 0.0)

    def _query_embedding(
        self,
        text: str,
        *,
        purpose: EmbeddingPurpose,
    ) -> tuple[float, ...]:
        cache_key = (purpose, text)
        cached = self._query_embedding_cache.get(cache_key)
        if cached is not None:
            self._record_cache_hit("query_embedding")
            return cached
        self._record_cache_miss("query_embedding")
        if self.embedding_provider is None:
            return ()
        vector = self.embedding_provider.embed_texts([text], purpose=purpose)[0]
        self._bounded_cache_store("query_embedding", self._query_embedding_cache, cache_key, vector)
        return vector

    def _lexical_seed_support(
        self,
        text: str,
        *,
        seed_phrases: tuple[str, ...],
        negative_phrases: tuple[str, ...],
    ) -> float:
        normalized = normalize_alias(text)
        if not normalized:
            return 0.0
        positive_hits = sum(1 for phrase in seed_phrases if phrase and normalize_alias(phrase) in normalized)
        negative_hits = sum(1 for phrase in negative_phrases if phrase and normalize_alias(phrase) in normalized)
        if positive_hits <= 0:
            return 0.0
        score = min(0.42 + (positive_hits * 0.18), 1.0)
        if negative_hits:
            score = max(score - (negative_hits * 0.24), 0.0)
        return round(score, 4)

    def _rank_aspect_candidates(
        self,
        *,
        window_text: str,
        target: TargetReference,
    ) -> list[PrototypeCandidateScore]:
        if self.prototype_registry is None:
            return []
        ranked: list[PrototypeCandidateScore] = []
        for aspect in self.prototype_registry.aspects.values():
            if aspect.compatible_entity_types and (target.entity_type or "") not in aspect.compatible_entity_types:
                continue
            semantic = self._prototype_similarity(
                window_text,
                aspect.prototype_text,
                prototype_vector=aspect.vector,
            )
            lexical = self._lexical_seed_support(
                window_text,
                seed_phrases=aspect.seed_phrases,
                negative_phrases=aspect.negative_phrases,
            )
            score = round(
                (semantic * 0.34)
                + (lexical * 0.46)
                + (max(semantic, lexical) * 0.2),
                6,
            )
            if score >= 0.22:
                ranked.append(
                    PrototypeCandidateScore(
                        label=aspect.aspect_id,
                        total_score=score,
                        lexical_score=round(lexical, 6),
                        semantic_score=round(semantic, 6),
                    )
                )
        return sorted(ranked, key=lambda item: (-item.total_score, item.label))[:4]

    def _rank_issue_candidates(
        self,
        *,
        window_text: str,
        target: TargetReference,
    ) -> list[PrototypeCandidateScore]:
        if self.prototype_registry is None:
            return []
        ranked: list[PrototypeCandidateScore] = []
        for issue in self.prototype_registry.issues.values():
            if issue.compatible_entity_types and (target.entity_type or "") not in issue.compatible_entity_types:
                continue
            semantic = self._prototype_similarity(
                window_text,
                issue.prototype_text,
                prototype_vector=issue.vector,
            )
            lexical = self._lexical_seed_support(
                window_text,
                seed_phrases=issue.seed_phrases,
                negative_phrases=issue.negative_phrases,
            )
            score = round(
                (semantic * 0.48)
                + (lexical * 0.32)
                + (max(semantic, lexical) * 0.2),
                6,
            )
            if score >= 0.22:
                ranked.append(
                    PrototypeCandidateScore(
                        label=issue.issue_id,
                        total_score=score,
                        lexical_score=round(lexical, 6),
                        semantic_score=round(semantic, 6),
                    )
                )
        return sorted(ranked, key=lambda item: (-item.total_score, item.label))[:4]

    def _related_issues_for_aspect(self, aspect_id: str) -> tuple[str, ...]:
        if self.prototype_registry is None:
            mapped_issue = _ASPECT_TO_ISSUE.get(aspect_id)
            return (mapped_issue,) if mapped_issue is not None else ()
        aspect_bundle = self.prototype_registry.aspects.get(aspect_id)
        if aspect_bundle is None or not aspect_bundle.related_issue_ids:
            mapped_issue = _ASPECT_TO_ISSUE.get(aspect_id)
            if mapped_issue is None or mapped_issue not in self.prototype_registry.issues:
                return ()
            return (mapped_issue,)
        primary_issue = aspect_bundle.related_issue_ids[0]
        return (primary_issue,)

    def _segment_aspect_opinions(
        self,
        *,
        mention: MentionRecord,
        segment: SemanticSegment,
        anchors: list[SemanticAnchor],
        explicit_targets: list[TargetReference],
        inherited_targets: list[TargetReference],
        uncertainty_flags: list[str],
        semantic_supports: dict[str, list[SemanticAssistMatch]],
        routing_mode: str,
    ) -> list[AspectOpinionHypothesis]:
        if not any(anchor.anchor_type == AnchorType.POLARITY for anchor in anchors):
            return []
        results: list[AspectOpinionHypothesis] = []
        grounded_targets = self._grounded_targets_for_segment(
            anchors=anchors,
            explicit_targets=explicit_targets,
            inherited_targets=inherited_targets,
        )
        if not grounded_targets:
            return []
        aspect_anchor_map: dict[str, list[SemanticAnchor]] = defaultdict(list)
        for anchor in anchors:
            if anchor.anchor_type == AnchorType.ASPECT:
                aspect_anchor_map[anchor.label].append(anchor)
        max_inferred_candidates = 2 if aspect_anchor_map else 1
        inferred_candidates_emitted = 0
        for target_ref, evidence_scope, target_anchor in grounded_targets:
            window_text = self._target_window_text(segment, target_anchor)
            for candidate_score in self._rank_aspect_candidates(
                window_text=window_text,
                target=target_ref,
            ):
                aspect_label = candidate_score.label
                semantic_score = candidate_score.total_score
                matching_aspect_anchors = aspect_anchor_map.get(aspect_label, [])
                support_matches = semantic_supports.get(aspect_label, [])
                lexically_supported = candidate_score.lexical_score >= 0.18
                inferred_only = not matching_aspect_anchors and not support_matches and not lexically_supported
                if inferred_only and (
                    candidate_score.semantic_score < 0.6
                    or inferred_candidates_emitted >= max_inferred_candidates
                ):
                    continue
                if matching_aspect_anchors:
                    local_score, supporting_cues = self._score_cues_for_anchor(matching_aspect_anchors[0], anchors)
                    leading_evidence = [matching_aspect_anchors[0].to_evidence_span()]
                else:
                    local_score, supporting_cues = self._score_cues_for_target(
                        target=target_ref,
                        anchors=anchors,
                        explicit_anchor=target_anchor is not None,
                    )
                    leading_evidence = [target_anchor.to_evidence_span()] if target_anchor is not None else []
                if not supporting_cues or abs(local_score) < 0.18:
                    continue
                prototype_component = ScoreComponent(
                    name="aspect_alignment_score",
                    value=semantic_score,
                    reason=f"target-conditioned aspect alignment score for {aspect_label}",
                )
                components = sentiment_confidence_components(
                    explicit_target=target_ref is not None and not target_ref.inherited and evidence_scope == EvidenceScope.LOCAL,
                    inherited_target=target_ref.inherited if target_ref else False,
                    support_count=len(supporting_cues),
                    mixed_evidence=self._sentiment_label(local_score) == "mixed",
                    contrastive=segment.contrastive,
                    negated=any(anchor.anchor_type == AnchorType.NEGATION for anchor in anchors),
                    uncertain=bool(uncertainty_flags),
                )
                components.append(prototype_component)
                components.extend(self._semantic_support_components(aspect_label, support_matches))
                confidence = self._apply_semantic_support_bonus(
                    clamp_confidence(0.48, components),
                    support_matches,
                )
                evidence_spans = [*leading_evidence, *[cue.to_evidence_span() for cue in supporting_cues]]
                results.append(
                    AspectOpinionHypothesis(
                        mention_id=mention.mention_id,
                        aspect=aspect_label,
                        target=target_ref,
                        sentiment=self._sentiment_label(local_score),
                        confidence=confidence,
                        semantic_routing=routing_mode,
                        target_grounding_confidence=target_ref.grounding_confidence,
                        corroboration_confidence=0.62,
                        evidence_scope=evidence_scope,
                        evidence_spans=evidence_spans,
                        score_components=components,
                        uncertainty_flags=uncertainty_flags,
                        segment_id=segment.segment_id,
                    )
                )
                if inferred_only:
                    inferred_candidates_emitted += 1
        return results

    def _segment_issue_signals(
        self,
        *,
        mention: MentionRecord,
        segment: SemanticSegment,
        anchors: list[SemanticAnchor],
        explicit_targets: list[TargetReference],
        inherited_targets: list[TargetReference],
        uncertainty_flags: list[str],
        semantic_supports: dict[str, list[SemanticAssistMatch]],
        routing_mode: str,
    ) -> list[IssueSignalHypothesis]:
        grounded_targets = self._grounded_targets_for_segment(
            anchors=anchors,
            explicit_targets=explicit_targets,
            inherited_targets=inherited_targets,
        )
        if not grounded_targets:
            return []
        results: list[IssueSignalHypothesis] = []
        issue_anchor_map: dict[str, list[SemanticAnchor]] = defaultdict(list)
        aspect_anchor_map: dict[str, list[SemanticAnchor]] = defaultdict(list)
        for anchor in anchors:
            if anchor.anchor_type == AnchorType.ISSUE:
                issue_anchor_map[anchor.label].append(anchor)
            if anchor.anchor_type == AnchorType.ASPECT:
                aspect_anchor_map[anchor.label].append(anchor)
        for target_ref, evidence_scope, target_anchor in grounded_targets:
            window_text = self._target_window_text(segment, target_anchor)
            ranked_issues = {
                candidate.label: candidate
                for candidate in self._rank_issue_candidates(
                    window_text=window_text,
                    target=target_ref,
                )
            }
            for aspect_label, anchors_for_aspect in aspect_anchor_map.items():
                local_score, _ = self._score_cues_for_anchor(anchors_for_aspect[0], anchors)
                if local_score < -0.2:
                    for issue_label in self._related_issues_for_aspect(aspect_label):
                        existing = ranked_issues.get(issue_label)
                        derived = PrototypeCandidateScore(
                            label=issue_label,
                            total_score=max(existing.total_score if existing is not None else 0.0, 0.44),
                            lexical_score=existing.lexical_score if existing is not None else 0.0,
                            semantic_score=existing.semantic_score if existing is not None else 0.62,
                        )
                        ranked_issues[issue_label] = derived
            for candidate_score in sorted(ranked_issues.values(), key=lambda item: (-item.total_score, item.label))[:4]:
                issue_label = candidate_score.label
                semantic_score = candidate_score.total_score
                explicit_issue_anchors = issue_anchor_map.get(issue_label, [])
                support_matches = semantic_supports.get(issue_label, [])
                lexically_supported = candidate_score.lexical_score >= 0.18
                if not explicit_issue_anchors and not support_matches and not lexically_supported and candidate_score.semantic_score < 0.62:
                    continue
                if explicit_issue_anchors:
                    score, polarity_support = self._score_cues_for_anchor(explicit_issue_anchors[0], anchors)
                    leading_evidence = [explicit_issue_anchors[0].to_evidence_span()]
                else:
                    score, polarity_support = self._score_cues_for_target(
                        target=target_ref,
                        anchors=anchors,
                        explicit_anchor=target_anchor is not None,
                    )
                    leading_evidence = [target_anchor.to_evidence_span()] if target_anchor is not None else []
                if score >= -0.1 and not explicit_issue_anchors:
                    continue
                evidence_mode = self._issue_evidence_mode(segment, anchors, score)
                severity = self._issue_severity(issue_label, evidence_mode, score)
                prototype_component = ScoreComponent(
                    name="issue_alignment_score",
                    value=semantic_score,
                    reason=f"target-conditioned issue alignment score for {issue_label}",
                )
                components = issue_confidence_components(
                    explicit_target=not target_ref.inherited and evidence_scope == EvidenceScope.LOCAL,
                    inherited_target=target_ref.inherited,
                    corroboration_count=1,
                    uncertain=bool(uncertainty_flags),
                    escalation=evidence_mode == EvidenceMode.ESCALATION_SIGNAL,
                    direct_mode=evidence_mode in {EvidenceMode.DIRECT_COMPLAINT, EvidenceMode.DIRECT_OBSERVATION},
                )
                components.append(prototype_component)
                components.extend(self._semantic_support_components(issue_label, support_matches))
                results.append(
                    IssueSignalHypothesis(
                        mention_id=mention.mention_id,
                        issue_category=issue_label,
                        target=target_ref,
                        severity=severity,
                        evidence_mode=evidence_mode,
                        confidence=self._apply_semantic_support_bonus(
                            clamp_confidence(0.46, components),
                            support_matches,
                        ),
                        semantic_routing=routing_mode,
                        target_grounding_confidence=target_ref.grounding_confidence,
                        corroboration_confidence=0.58,
                        evidence_scope=evidence_scope,
                        evidence_spans=[*leading_evidence, *[anchor.to_evidence_span() for anchor in polarity_support]],
                        score_components=components,
                        uncertainty_flags=uncertainty_flags,
                        segment_id=segment.segment_id,
                    )
                )
        return results

    def _score_cues_for_anchor(
        self,
        anchor: SemanticAnchor,
        anchors: list[SemanticAnchor],
    ) -> tuple[float, list[SemanticAnchor]]:
        polarity_anchors = [candidate for candidate in anchors if candidate.anchor_type == AnchorType.POLARITY]
        weighted: list[tuple[SemanticAnchor, float]] = []
        for cue in polarity_anchors:
            distance = min(abs(cue.start - anchor.start), abs(cue.end - anchor.end))
            if distance > _SCOPE_MATCH_CHARS:
                continue
            weight = max(0.35, 1.0 - (distance / _SCOPE_MATCH_CHARS))
            weighted.append((cue, self._cue_score(cue, anchors) * weight))
        if not weighted:
            return 0.0, []
        return sum(score for _, score in weighted), [cue for cue, _ in weighted]

    def _issue_evidence_mode(
        self,
        segment: SemanticSegment,
        anchors: list[SemanticAnchor],
        score: float,
    ) -> EvidenceMode:
        if any(anchor.anchor_type == AnchorType.ESCALATION for anchor in anchors):
            return EvidenceMode.ESCALATION_SIGNAL
        if any(anchor.anchor_type == AnchorType.HEARSAY for anchor in anchors):
            return EvidenceMode.HEARSAY_OR_RUMOR
        if segment.question_like or any(anchor.anchor_type == AnchorType.UNCERTAINTY for anchor in anchors):
            return EvidenceMode.QUESTION_OR_UNCERTAINTY
        if any(anchor.anchor_type == AnchorType.COMPARISON for anchor in anchors) and score < 0:
            return EvidenceMode.COMPARISON_BASED_CRITIQUE
        if score < 0 and contains_first_person(segment):
            return EvidenceMode.DIRECT_COMPLAINT
        return EvidenceMode.DIRECT_OBSERVATION

    def _issue_severity(
        self,
        issue_label: str,
        evidence_mode: EvidenceMode,
        score: float,
    ) -> IssueSeverity:
        magnitude = abs(score)
        if issue_label == "safety_concern" and evidence_mode in {
            EvidenceMode.ESCALATION_SIGNAL,
            EvidenceMode.DIRECT_COMPLAINT,
        }:
            return "critical_like_proxy"
        if evidence_mode == EvidenceMode.ESCALATION_SIGNAL or magnitude >= 1.05:
            return "high"
        if evidence_mode in {EvidenceMode.QUESTION_OR_UNCERTAINTY, EvidenceMode.HEARSAY_OR_RUMOR}:
            return "low"
        if magnitude >= 0.55:
            return "medium"
        return "low"

    def _cue_score(self, cue: SemanticAnchor, anchors: list[SemanticAnchor]) -> float:
        polarity = cue.polarity or 0.0
        negation_anchors = [anchor for anchor in anchors if anchor.anchor_type == AnchorType.NEGATION]
        if any(0 <= cue.start - neg.start <= 14 for neg in negation_anchors):
            return -polarity
        return polarity

    def _aggregate_mention_sentiment(
        self,
        mention: MentionRecord,
        segment_sentiments: list[tuple[SemanticSegment, float, list[SemanticAnchor], list[str], bool]],
    ) -> MentionSentimentHypothesis | None:
        if not segment_sentiments:
            return None
        total_score = sum(score for _, score, _, _, _ in segment_sentiments)
        segment_labels = [self._sentiment_label(score) for _, score, _, _, _ in segment_sentiments]
        evidence_spans = [
            anchor.to_evidence_span()
            for _, _, anchors, _, _ in segment_sentiments
            for anchor in anchors
            if anchor.anchor_type == AnchorType.POLARITY
        ]
        uncertainty_flags = sorted({flag for _, _, _, flags, _ in segment_sentiments for flag in flags})
        sentiment = self._merge_sentiment_labels(segment_labels)
        components = sentiment_confidence_components(
            explicit_target=False,
            inherited_target=False,
            support_count=len(evidence_spans),
            mixed_evidence=sentiment == "mixed",
            contrastive=any(segment.contrastive for segment, _, _, _, _ in segment_sentiments),
            negated=any(negated for _, _, _, _, negated in segment_sentiments),
            uncertain=bool(uncertainty_flags),
        )
        confidence = clamp_confidence(0.5, components)
        return MentionSentimentHypothesis(
            mention_id=mention.mention_id,
            sentiment=sentiment,
            confidence=confidence,
            score=round(abs(total_score), 3),
            semantic_routing=self._semantic_routing_mode(mention),
            corroboration_confidence=0.65,
            evidence_spans=evidence_spans,
            score_components=components,
            uncertainty_flags=uncertainty_flags,
            segment_ids=[segment.segment_id for segment, _, _, _, _ in segment_sentiments],
        )

    def _merge_target_hypotheses(
        self,
        hypotheses: list[TargetSentimentHypothesis],
    ) -> list[TargetSentimentHypothesis]:
        grouped: dict[tuple[str, str], list[TargetSentimentHypothesis]] = defaultdict(list)
        for hypothesis in hypotheses:
            grouped[(hypothesis.mention_id, hypothesis.target.target_key)].append(hypothesis)
        merged: list[TargetSentimentHypothesis] = []
        for items in grouped.values():
            merged_target = self._merge_target_reference(items)
            merged.append(
                TargetSentimentHypothesis(
                    mention_id=items[0].mention_id,
                    target=merged_target,
                    sentiment=self._merge_sentiment_labels([item.sentiment for item in items]),
                    confidence=round(sum(item.confidence for item in items) / len(items), 3),
                    score=round(max(item.score for item in items), 3),
                    semantic_routing=items[0].semantic_routing,
                    target_grounding_confidence=max(
                        item.target_grounding_confidence or 0.0 for item in items
                    ),
                    corroboration_confidence=max(item.corroboration_confidence or 0.0 for item in items),
                    evidence_scope=self._merge_evidence_scope([item.evidence_scope for item in items]),
                    evidence_spans=[span for item in items for span in item.evidence_spans],
                    score_components=[component for item in items for component in item.score_components],
                    uncertainty_flags=sorted({flag for item in items for flag in item.uncertainty_flags}),
                    segment_ids=sorted({segment_id for item in items for segment_id in item.segment_ids}),
                )
            )
        return merged

    def _merge_target_reference(self, items: list[TargetSentimentHypothesis]) -> TargetReference:
        preferred = min(
            items,
            key=lambda item: (
                0 if item.target.inherited else 1,
                0 if item.evidence_scope == EvidenceScope.INHERITED else 1,
                0 if item.target.inherited_from_mention_id else 1,
                -len(item.evidence_spans),
                -item.confidence,
            ),
        )
        return preferred.target

    def _merge_aspect_hypotheses(
        self,
        hypotheses: list[AspectOpinionHypothesis],
    ) -> list[AspectOpinionHypothesis]:
        grouped: dict[tuple[str, str, str | None], list[AspectOpinionHypothesis]] = defaultdict(list)
        for hypothesis in hypotheses:
            grouped[
                (
                    hypothesis.mention_id,
                    hypothesis.aspect,
                    hypothesis.target.target_key if hypothesis.target is not None else None,
                )
            ].append(hypothesis)
        merged: list[AspectOpinionHypothesis] = []
        for items in grouped.values():
            first = items[0]
            merged.append(
                AspectOpinionHypothesis(
                    mention_id=first.mention_id,
                    aspect=first.aspect,
                    target=first.target,
                    sentiment=self._merge_sentiment_labels([item.sentiment for item in items]),
                    confidence=round(sum(item.confidence for item in items) / len(items), 3),
                    semantic_routing=first.semantic_routing,
                    target_grounding_confidence=max(
                        item.target_grounding_confidence or 0.0 for item in items
                    ),
                    corroboration_confidence=max(item.corroboration_confidence or 0.0 for item in items),
                    evidence_scope=self._merge_evidence_scope([item.evidence_scope for item in items]),
                    evidence_spans=[span for item in items for span in item.evidence_spans],
                    score_components=[component for item in items for component in item.score_components],
                    uncertainty_flags=sorted({flag for item in items for flag in item.uncertainty_flags}),
                    segment_id=first.segment_id,
                )
            )
        return merged

    def _rerank_aspect_hypotheses(
        self,
        *,
        mentions: dict[str, MentionRecord],
        hypotheses: list[AspectOpinionHypothesis],
    ) -> list[AspectOpinionHypothesis]:
        if self.taxonomy_mapping_provider is None:
            return hypotheses
        grouped: dict[tuple[str, str, str | None], list[AspectOpinionHypothesis]] = defaultdict(list)
        for hypothesis in hypotheses:
            grouped[
                (
                    hypothesis.mention_id,
                    hypothesis.segment_id,
                    hypothesis.target.target_key if hypothesis.target is not None else None,
                )
            ].append(hypothesis)
        updated: list[AspectOpinionHypothesis] = []
        for group in grouped.values():
            if len(group) == 1:
                updated.extend(group)
                continue
            mention = mentions[group[0].mention_id]
            query_text = self._hypothesis_query_text(group[0].segment_id, mention, [item.evidence_spans for item in group])
            if group[0].target is not None:
                query_text = f"{group[0].target.target_text} || {query_text}"
            taxonomy_labels = tuple(sorted({item.aspect for item in group}))
            score_by_label = self._taxonomy_rerank_scores(query_text, taxonomy_labels=taxonomy_labels, top_k=len(group))
            top_score = max(score_by_label.values(), default=0.0)
            for hypothesis in group:
                support = score_by_label.get(hypothesis.aspect, 0.0)
                confidence = hypothesis.confidence
                components = list(hypothesis.score_components)
                if support > 0.0:
                    components.append(
                        ScoreComponent(
                            name="semantic_hypothesis_rerank",
                            value=round(support, 4),
                            reason=f"semantic rerank support for aspect '{hypothesis.aspect}'",
                        )
                    )
                    confidence = round(min(confidence + min(support * 0.06, 0.08), 0.97), 3)
                    if top_score - support >= 0.12:
                        confidence = round(max(confidence - 0.08, 0.0), 3)
                updated.append(
                    AspectOpinionHypothesis(
                        mention_id=hypothesis.mention_id,
                        aspect=hypothesis.aspect,
                        target=hypothesis.target,
                        sentiment=hypothesis.sentiment,
                        confidence=confidence,
                        evidence_scope=hypothesis.evidence_scope,
                        evidence_spans=hypothesis.evidence_spans,
                        score_components=components,
                        uncertainty_flags=hypothesis.uncertainty_flags,
                        segment_id=hypothesis.segment_id,
                    )
                )
        return updated

    def _rerank_issue_hypotheses(
        self,
        *,
        mentions: dict[str, MentionRecord],
        hypotheses: list[IssueSignalHypothesis],
    ) -> list[IssueSignalHypothesis]:
        if self.taxonomy_mapping_provider is None:
            return hypotheses
        grouped: dict[tuple[str, str, str | None], list[IssueSignalHypothesis]] = defaultdict(list)
        for hypothesis in hypotheses:
            grouped[
                (
                    hypothesis.mention_id,
                    hypothesis.segment_id,
                    hypothesis.target.target_key if hypothesis.target is not None else None,
                )
            ].append(hypothesis)
        updated: list[IssueSignalHypothesis] = []
        for group in grouped.values():
            mention = mentions[group[0].mention_id]
            query_text = self._hypothesis_query_text(group[0].segment_id, mention, [item.evidence_spans for item in group])
            if group[0].target is not None:
                query_text = f"{group[0].target.target_text} || {query_text}"
            score_by_label: dict[str, float] = {}
            if len(group) > 1:
                score_by_label = self._taxonomy_rerank_scores(
                    query_text,
                    taxonomy_labels=tuple(sorted({item.issue_category for item in group})),
                    top_k=len(group),
                )
            mode_supports = self._rerank_issue_modes(query_text, group)
            top_issue_score = max(score_by_label.values(), default=0.0)
            for hypothesis in group:
                issue_support = score_by_label.get(hypothesis.issue_category, 0.0)
                mode_support = mode_supports.get(hypothesis.evidence_mode, 0.0)
                confidence = hypothesis.confidence
                components = list(hypothesis.score_components)
                if issue_support > 0.0:
                    components.append(
                        ScoreComponent(
                            name="semantic_issue_rerank",
                            value=round(issue_support, 4),
                            reason=f"semantic rerank support for issue '{hypothesis.issue_category}'",
                        )
                    )
                    confidence = round(min(confidence + min(issue_support * 0.05, 0.07), 0.97), 3)
                    if top_issue_score - issue_support >= 0.12:
                        confidence = round(max(confidence - 0.07, 0.0), 3)
                if mode_support > 0.0:
                    components.append(
                        ScoreComponent(
                            name="evidence_mode_rerank",
                            value=round(mode_support, 4),
                            reason=f"semantic rerank support for evidence mode '{hypothesis.evidence_mode.value}'",
                        )
                    )
                    confidence = round(min(confidence + min(mode_support * 0.04, 0.05), 0.97), 3)
                updated.append(
                    IssueSignalHypothesis(
                        mention_id=hypothesis.mention_id,
                        issue_category=hypothesis.issue_category,
                        target=hypothesis.target,
                        severity=hypothesis.severity,
                        evidence_mode=hypothesis.evidence_mode,
                        confidence=confidence,
                        evidence_scope=hypothesis.evidence_scope,
                        evidence_spans=hypothesis.evidence_spans,
                        score_components=components,
                        uncertainty_flags=hypothesis.uncertainty_flags,
                        segment_id=hypothesis.segment_id,
                        corroboration_count=hypothesis.corroboration_count,
                    )
                )
        return updated

    def _rerank_issue_modes(
        self,
        query_text: str,
        hypotheses: list[IssueSignalHypothesis],
    ) -> dict[EvidenceMode, float]:
        if self.reranker_provider is None:
            return {}
        cache_key = (query_text, tuple(sorted(item.evidence_mode.value for item in hypotheses)))
        cached = self._issue_mode_rerank_cache.get(cache_key)
        if cached is not None:
            self._record_cache_hit("issue_mode_rerank")
            return cached
        self._record_cache_miss("issue_mode_rerank")
        self._increment_counter("issue_mode_rerank_calls")
        descriptors: dict[EvidenceMode, str] = {
            EvidenceMode.DIRECT_COMPLAINT: "direct complaint firsthand problem not working unhappy",
            EvidenceMode.DIRECT_OBSERVATION: "direct observation saw happened experienced",
            EvidenceMode.QUESTION_OR_UNCERTAINTY: "question uncertainty maybe ask not sure",
            EvidenceMode.HEARSAY_OR_RUMOR: "hearsay rumor heard people say",
            EvidenceMode.COMPARISON_BASED_CRITIQUE: "comparison critique worse than compared",
            EvidenceMode.ESCALATION_SIGNAL: "escalation urgent refund report scam",
        }
        candidate_texts = [descriptors[item.evidence_mode] for item in hypotheses]
        started_at = perf_counter() if self._runtime_diagnostics_enabled else None
        ranked = self.reranker_provider.rerank(query_text, candidate_texts)
        if started_at is not None:
            self._record_timing("issue_mode_rerank", perf_counter() - started_at)
        score_by_text = {match.candidate_id: match.score for match in ranked}
        result = {
            hypothesis.evidence_mode: score_by_text.get(descriptors[hypothesis.evidence_mode], 0.0)
            for hypothesis in hypotheses
        }
        self._bounded_cache_store("issue_mode_rerank", self._issue_mode_rerank_cache, cache_key, result)
        return result

    def _hypothesis_query_text(
        self,
        segment_id: str | None,
        mention: MentionRecord,
        evidence_groups: list[list[EvidenceSpan]],
    ) -> str:
        evidence_text = " ".join(
            span.text
            for evidence in evidence_groups
            for span in evidence
            if span.text.strip()
        ).strip()
        if segment_id is None or not evidence_text:
            return f"{mention.raw_text} || {evidence_text}".strip(" |")
        return f"{segment_id} || {mention.raw_text} || {evidence_text}"

    def _apply_thread_corroboration(
        self,
        *,
        mentions: dict[str, MentionRecord],
        issue_hypotheses: list[IssueSignalHypothesis],
    ) -> list[IssueSignalHypothesis]:
        grouped: dict[tuple[str, str, str | None], list[IssueSignalHypothesis]] = defaultdict(list)
        for issue in issue_hypotheses:
            mention = mentions[issue.mention_id]
            grouped[(mention.root_id, issue.issue_category, issue.target.target_key if issue.target else None)].append(issue)
        corroboration_counts: dict[int, int] = {}
        for group in grouped.values():
            self._increment_counter("pairwise_similarity_group_count")
            self._increment_counter("pairwise_similarity_group_item_total", len(group))
            self._set_counter_max("pairwise_similarity_group_max_size", len(group))
            if len(group) > _CORROBORATION_PAIRWISE_MAX_GROUP:
                corroboration_counts.update(
                    self._large_group_corroboration_counts(
                        evidence_texts=[
                            self._issue_hypothesis_evidence_text(issue, mentions[issue.mention_id])
                            for issue in group
                        ],
                        hypotheses=group,
                    )
                )
                continue
            evidence_texts = [
                self._issue_hypothesis_evidence_text(issue, mentions[issue.mention_id])
                for issue in group
            ]
            similarities = self._pairwise_issue_similarity(evidence_texts)
            similarity_matrix = np.asarray(similarities, dtype=np.float32)
            corroboration_vector = (similarity_matrix >= _CORROBORATION_SIMILARITY_THRESHOLD).sum(axis=1)
            for index, issue in enumerate(group):
                corroboration_counts[id(issue)] = int(corroboration_vector[index])
        updated: list[IssueSignalHypothesis] = []
        for issue in issue_hypotheses:
            corroboration_count = corroboration_counts.get(id(issue), 1)
            components = issue_confidence_components(
                explicit_target=issue.target is not None and not issue.target.inherited and issue.evidence_scope == EvidenceScope.LOCAL,
                inherited_target=issue.target.inherited if issue.target else False,
                corroboration_count=corroboration_count,
                uncertain=bool(issue.uncertainty_flags),
                escalation=issue.evidence_mode == EvidenceMode.ESCALATION_SIGNAL,
                direct_mode=issue.evidence_mode in {EvidenceMode.DIRECT_COMPLAINT, EvidenceMode.DIRECT_OBSERVATION},
            )
            updated.append(
                IssueSignalHypothesis(
                    mention_id=issue.mention_id,
                    issue_category=issue.issue_category,
                    target=issue.target,
                    severity=self._bump_issue_severity(
                        issue.severity,
                        corroboration_count,
                        issue.evidence_mode,
                    ),
                    evidence_mode=issue.evidence_mode,
                    confidence=clamp_confidence(0.44, components),
                    evidence_scope=issue.evidence_scope,
                    evidence_spans=issue.evidence_spans,
                    score_components=components,
                    uncertainty_flags=issue.uncertainty_flags,
                    segment_id=issue.segment_id,
                    corroboration_count=corroboration_count,
                )
            )
        return updated

    def _issue_hypothesis_evidence_text(
        self,
        issue: IssueSignalHypothesis,
        mention: MentionRecord,
    ) -> str:
        if issue.evidence_spans:
            joined = " ".join(span.text for span in issue.evidence_spans if span.text.strip())
            if joined.strip():
                return joined
        return mention.raw_text

    def _pairwise_issue_similarity(
        self,
        evidence_texts: list[str],
    ) -> list[list[float]]:
        if not evidence_texts:
            return []
        cache_key = tuple(evidence_texts)
        cached = self._pairwise_similarity_cache.get(cache_key)
        if cached is not None:
            self._record_cache_hit("pairwise_similarity")
            return cached
        self._record_cache_miss("pairwise_similarity")
        started_at = perf_counter() if self._runtime_diagnostics_enabled else None
        if self.embedding_provider is None:
            normalized = [normalize_alias(text) for text in evidence_texts]
            result = np.eye(len(normalized), dtype=np.float32)
            for row in range(len(normalized)):
                for column in range(row + 1, len(normalized)):
                    score = (
                        1.0
                        if normalized[row] == normalized[column]
                        else self._token_jaccard(normalized[row], normalized[column])
                    )
                    result[row, column] = score
                    result[column, row] = score
            materialized = result.tolist()
            if started_at is not None:
                self._record_timing("pairwise_similarity", perf_counter() - started_at)
            self._bounded_cache_store("pairwise_similarity", self._pairwise_similarity_cache, cache_key, materialized)
            return materialized
        unique_texts = list(dict.fromkeys(evidence_texts))
        vector_map = dict(
            zip(
                unique_texts,
                self.embedding_provider.embed_texts(unique_texts, purpose=EmbeddingPurpose.PASSAGE),
                strict=True,
            )
        )
        vectors = np.asarray([vector_map[text] for text in evidence_texts], dtype=np.float32)
        similarities = np.matmul(vectors, vectors.T)
        materialized = np.round(similarities, 4).tolist()
        if started_at is not None:
            self._record_timing("pairwise_similarity", perf_counter() - started_at)
        self._bounded_cache_store("pairwise_similarity", self._pairwise_similarity_cache, cache_key, materialized)
        return materialized

    def _token_jaccard(self, left: str, right: str) -> float:
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens and not right_tokens:
            return 1.0
        union = left_tokens | right_tokens
        if not union:
            return 0.0
        return len(left_tokens & right_tokens) / len(union)

    def _semantic_support_components(
        self,
        label: str,
        support_matches: list[SemanticAssistMatch],
    ) -> list[ScoreComponent]:
        if not support_matches:
            return []
        best_match = support_matches[0]
        return [
            ScoreComponent(
                name="semantic_assist_support",
                value=round(min(best_match.score, 1.0), 4),
                reason=(
                    f"semantic assist mapped '{best_match.source_text}' to '{label}' "
                    f"via {best_match.provider_name}"
                ),
            ),
            ScoreComponent(
                name="semantic_assist_gap",
                value=round(min(best_match.ambiguity_gap, 1.0), 4),
                reason=f"semantic assist ambiguity gap for '{label}' was {best_match.ambiguity_gap:.3f}",
            ),
        ]

    def _apply_semantic_support_bonus(
        self,
        confidence: float,
        support_matches: list[SemanticAssistMatch],
    ) -> float:
        if not support_matches:
            return confidence
        strongest = max(match.score for match in support_matches)
        ambiguity_gap = max(match.ambiguity_gap for match in support_matches)
        bonus = min(0.10, max(0.0, strongest - _SEMANTIC_ASSIST_MIN_SCORE) * 0.25)
        if ambiguity_gap < 0.04:
            bonus = min(bonus, 0.02)
        return round(min(confidence + bonus, 0.96), 3)

    def _apply_aspect_corroboration(
        self,
        *,
        mentions: dict[str, MentionRecord],
        aspect_hypotheses: list[AspectOpinionHypothesis],
    ) -> list[AspectOpinionHypothesis]:
        grouped: dict[tuple[str, str, str | None, str], list[AspectOpinionHypothesis]] = defaultdict(list)
        for aspect in aspect_hypotheses:
            mention = mentions[aspect.mention_id]
            grouped[
                (
                    mention.root_id,
                    aspect.aspect,
                    aspect.target.target_key if aspect.target else None,
                    aspect.sentiment,
                )
            ].append(aspect)
        corroboration_counts: dict[int, int] = {}
        for group in grouped.values():
            self._increment_counter("pairwise_similarity_group_count")
            self._increment_counter("pairwise_similarity_group_item_total", len(group))
            self._set_counter_max("pairwise_similarity_group_max_size", len(group))
            if len(group) > _CORROBORATION_PAIRWISE_MAX_GROUP:
                corroboration_counts.update(
                    self._large_group_corroboration_counts(
                        evidence_texts=[
                            self._aspect_hypothesis_evidence_text(aspect, mentions[aspect.mention_id])
                            for aspect in group
                        ],
                        hypotheses=group,
                    )
                )
                continue
            evidence_texts = [
                self._aspect_hypothesis_evidence_text(aspect, mentions[aspect.mention_id])
                for aspect in group
            ]
            similarities = self._pairwise_issue_similarity(evidence_texts)
            similarity_matrix = np.asarray(similarities, dtype=np.float32)
            corroboration_vector = (similarity_matrix >= _CORROBORATION_SIMILARITY_THRESHOLD).sum(axis=1)
            for index, aspect in enumerate(group):
                corroboration_counts[id(aspect)] = int(corroboration_vector[index])
        updated: list[AspectOpinionHypothesis] = []
        for aspect in aspect_hypotheses:
            corroboration_count = corroboration_counts.get(id(aspect), 1)
            components = list(aspect.score_components)
            if corroboration_count > 1:
                components.append(
                    ScoreComponent(
                        name="semantic_corroboration",
                        value=round(min(corroboration_count / 4.0, 1.0), 4),
                        reason=f"thread-level semantic corroboration from {corroboration_count} similar aspect mentions",
                    )
                )
            updated.append(
                AspectOpinionHypothesis(
                    mention_id=aspect.mention_id,
                    aspect=aspect.aspect,
                    target=aspect.target,
                    sentiment=aspect.sentiment,
                    confidence=round(min(aspect.confidence + max(corroboration_count - 1, 0) * 0.03, 0.97), 3),
                    evidence_scope=aspect.evidence_scope,
                    evidence_spans=aspect.evidence_spans,
                    score_components=components,
                    uncertainty_flags=aspect.uncertainty_flags,
                    segment_id=aspect.segment_id,
                )
            )
        return updated

    def _aspect_hypothesis_evidence_text(
        self,
        aspect: AspectOpinionHypothesis,
        mention: MentionRecord,
    ) -> str:
        if aspect.evidence_spans:
            joined = " ".join(span.text for span in aspect.evidence_spans if span.text.strip())
            if joined.strip():
                return joined
        return mention.raw_text

    def _large_group_corroboration_counts(
        self,
        *,
        evidence_texts: list[str],
        hypotheses: Sequence[IssueSignalHypothesis | AspectOpinionHypothesis],
    ) -> dict[int, int]:
        normalized_counts: dict[str, int] = defaultdict(int)
        normalized_texts: list[str] = []
        for text in evidence_texts:
            normalized = normalize_alias(text) or text.casefold().strip()
            normalized_texts.append(normalized)
            normalized_counts[normalized] += 1
        return {
            id(hypothesis): max(1, normalized_counts[normalized_texts[index]])
            for index, hypothesis in enumerate(hypotheses)
        }

    def _bump_issue_severity(
        self,
        severity: str,
        corroboration_count: int,
        evidence_mode: EvidenceMode,
    ) -> IssueSeverity:
        if evidence_mode in {EvidenceMode.QUESTION_OR_UNCERTAINTY, EvidenceMode.HEARSAY_OR_RUMOR}:
            return "low"
        if corroboration_count <= 1:
            return cast(IssueSeverity, severity) if severity in {"low", "medium", "high", "critical_like_proxy"} else "low"
        if severity == "low":
            return "medium"
        if severity == "medium" and corroboration_count >= 3:
            return "high"
        return cast(IssueSeverity, severity) if severity in {"low", "medium", "high", "critical_like_proxy"} else "low"

    def _build_mention_sentiment_fact(self, hypothesis: MentionSentimentHypothesis, mention: MentionRecord) -> SentimentFact:
        return SentimentFact(
            mention_id=mention.mention_id,
            source_uap_id=mention.source_uap_id,
            sentiment=hypothesis.sentiment,  # type: ignore[arg-type]
            score=hypothesis.score,
            confidence=hypothesis.confidence,
            semantic_routing=hypothesis.semantic_routing,  # type: ignore[arg-type]
            sentiment_confidence=hypothesis.confidence,
            corroboration_confidence=hypothesis.corroboration_confidence,
            evidence_spans=hypothesis.evidence_spans,
            uncertainty_flags=hypothesis.uncertainty_flags,
            score_components=hypothesis.score_components,
            segment_ids=hypothesis.segment_ids,
            provenance=FactProvenance(
                source_uap_id=mention.source_uap_id,
                mention_id=mention.mention_id,
                provider_version=self.provider_version,
                rule_version="semantic-sentiment-v2",
                evidence_text=mention.raw_text,
            ),
        )

    def _build_target_sentiment_fact(
        self,
        hypothesis: TargetSentimentHypothesis,
        mention: MentionRecord,
    ) -> TargetSentimentFact:
        return TargetSentimentFact(
            mention_id=mention.mention_id,
            source_uap_id=mention.source_uap_id,
            target_key=hypothesis.target.target_key,
            target_text=hypothesis.target.target_text,
            canonical_entity_id=hypothesis.target.canonical_entity_id,
            concept_entity_id=hypothesis.target.concept_entity_id,
            unresolved_cluster_id=hypothesis.target.unresolved_cluster_id,
            target_kind=hypothesis.target.target_kind,  # type: ignore[arg-type]
            entity_type=hypothesis.target.entity_type,
            sentiment=hypothesis.sentiment,  # type: ignore[arg-type]
            score=hypothesis.score,
            confidence=hypothesis.confidence,
            semantic_routing=hypothesis.semantic_routing,  # type: ignore[arg-type]
            sentiment_confidence=hypothesis.confidence,
            target_grounding_confidence=hypothesis.target_grounding_confidence,
            corroboration_confidence=hypothesis.corroboration_confidence,
            target_inherited=hypothesis.target.inherited,
            inherited_from_mention_id=hypothesis.target.inherited_from_mention_id,
            evidence_scope=hypothesis.evidence_scope,
            evidence_spans=hypothesis.evidence_spans,
            uncertainty_flags=hypothesis.uncertainty_flags,
            score_components=hypothesis.score_components,
            segment_ids=hypothesis.segment_ids,
            provenance=FactProvenance(
                source_uap_id=mention.source_uap_id,
                mention_id=mention.mention_id,
                provider_version=self.provider_version,
                rule_version="semantic-target-sentiment-v2",
                evidence_text=mention.raw_text,
            ),
        )

    def _build_aspect_fact(self, hypothesis: AspectOpinionHypothesis, mention: MentionRecord) -> AspectOpinionFact:
        opinion_text = mention.raw_text
        if hypothesis.evidence_spans:
            start = min(span.start for span in hypothesis.evidence_spans)
            end = max(span.end for span in hypothesis.evidence_spans)
            opinion_text = mention.raw_text[start:end]
        return AspectOpinionFact(
            mention_id=mention.mention_id,
            source_uap_id=mention.source_uap_id,
            aspect=hypothesis.aspect,
            opinion_text=opinion_text,
            sentiment=hypothesis.sentiment,
            confidence=hypothesis.confidence,
            target_key=hypothesis.target.target_key if hypothesis.target else None,
            canonical_entity_id=hypothesis.target.canonical_entity_id if hypothesis.target else None,
            concept_entity_id=hypothesis.target.concept_entity_id if hypothesis.target else None,
            unresolved_cluster_id=hypothesis.target.unresolved_cluster_id if hypothesis.target else None,
            target_kind=(hypothesis.target.target_kind if hypothesis.target else "surface"),  # type: ignore[arg-type]
            semantic_routing=hypothesis.semantic_routing,  # type: ignore[arg-type]
            sentiment_confidence=hypothesis.confidence,
            target_grounding_confidence=hypothesis.target_grounding_confidence,
            corroboration_confidence=hypothesis.corroboration_confidence,
            target_inherited=hypothesis.target.inherited if hypothesis.target else False,
            inherited_from_mention_id=hypothesis.target.inherited_from_mention_id if hypothesis.target else None,
            evidence_scope=hypothesis.evidence_scope,
            evidence_spans=hypothesis.evidence_spans,
            uncertainty_flags=hypothesis.uncertainty_flags,
            score_components=hypothesis.score_components,
            segment_id=hypothesis.segment_id,
            provenance=FactProvenance(
                source_uap_id=mention.source_uap_id,
                mention_id=mention.mention_id,
                provider_version=self.provider_version,
                rule_version="semantic-aspect-opinion-v2",
                evidence_text=mention.raw_text,
            ),
        )

    def _build_issue_fact(self, hypothesis: IssueSignalHypothesis, mention: MentionRecord) -> IssueSignalFact:
        return IssueSignalFact(
            mention_id=mention.mention_id,
            source_uap_id=mention.source_uap_id,
            issue_category=hypothesis.issue_category,
            severity=hypothesis.severity,  # type: ignore[arg-type]
            confidence=hypothesis.confidence,
            evidence_mode=hypothesis.evidence_mode,
            target_key=hypothesis.target.target_key if hypothesis.target else None,
            canonical_entity_id=hypothesis.target.canonical_entity_id if hypothesis.target else None,
            concept_entity_id=hypothesis.target.concept_entity_id if hypothesis.target else None,
            unresolved_cluster_id=hypothesis.target.unresolved_cluster_id if hypothesis.target else None,
            target_kind=(hypothesis.target.target_kind if hypothesis.target else "surface"),  # type: ignore[arg-type]
            semantic_routing=hypothesis.semantic_routing,  # type: ignore[arg-type]
            issue_evidence_confidence=hypothesis.confidence,
            target_grounding_confidence=hypothesis.target_grounding_confidence,
            corroboration_confidence=hypothesis.corroboration_confidence,
            target_inherited=hypothesis.target.inherited if hypothesis.target else False,
            inherited_from_mention_id=hypothesis.target.inherited_from_mention_id if hypothesis.target else None,
            evidence_scope=hypothesis.evidence_scope,
            evidence_spans=hypothesis.evidence_spans,
            uncertainty_flags=hypothesis.uncertainty_flags,
            score_components=hypothesis.score_components,
            corroboration_count=hypothesis.corroboration_count,
            segment_id=hypothesis.segment_id,
            provenance=FactProvenance(
                source_uap_id=mention.source_uap_id,
                mention_id=mention.mention_id,
                provider_version=self.provider_version,
                rule_version="semantic-issue-signal-v2",
                evidence_text=mention.raw_text,
            ),
        )

    def _sentiment_label(self, score: float) -> SentimentValue:
        if score >= 0.35:
            return "positive"
        if score <= -0.35:
            return "negative"
        if abs(score) < 0.15:
            return "neutral"
        return "mixed"

    def _merge_sentiment_labels(self, labels: Sequence[str]) -> SentimentValue:
        label_set = set(labels)
        if {"positive", "negative"} <= label_set or "mixed" in label_set:
            return "mixed"
        if "positive" in label_set:
            return "positive"
        if "negative" in label_set:
            return "negative"
        return "neutral"

    def _merge_evidence_scope(self, scopes: list[EvidenceScope]) -> EvidenceScope:
        if EvidenceScope.LOCAL in scopes:
            return EvidenceScope.LOCAL
        if EvidenceScope.INHERITED in scopes:
            return EvidenceScope.INHERITED
        return EvidenceScope.AMBIGUOUS
