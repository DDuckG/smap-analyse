from __future__ import annotations

import atexit
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from time import perf_counter

from smap.core.settings import Settings
from smap.enrichers.semantic_models import SemanticHypothesisBatch, TargetReference
from smap.normalization.models import MentionRecord
from smap.ontology.models import OntologyRegistry
from smap.ontology.prototypes import PrototypeRegistry
from smap.providers.factory import SemanticProviderRuntime, build_semantic_provider_runtime
from smap.threads.models import MentionContext


@dataclass(slots=True)
class _SemanticWorkerState:
    runtime: SemanticProviderRuntime
    semantic_enricher: object


_WORKER_STATE: _SemanticWorkerState | None = None


def _close_worker_resources() -> None:
    global _WORKER_STATE
    if _WORKER_STATE is None:
        return
    resources = [
        _WORKER_STATE.runtime.embedding_provider,
        _WORKER_STATE.runtime.taxonomy_mapping_provider,
        _WORKER_STATE.runtime.reranker_provider,
        getattr(_WORKER_STATE.runtime.embedding_provider, "cache_store", None),
    ]
    seen: set[int] = set()
    for resource in resources:
        close_method = getattr(resource, "close", None)
        if resource is None or not callable(close_method) or id(resource) in seen:
            continue
        seen.add(id(resource))
        close_method()
    _WORKER_STATE = None


def _init_worker(settings_payload: dict[str, object], ontology_payload: dict[str, object]) -> None:
    global _WORKER_STATE

    from smap.enrichers.semantic import SemanticInferenceEnricher

    settings = Settings.model_validate(settings_payload)
    settings.ensure_directories()
    ontology = OntologyRegistry.model_validate(ontology_payload)
    runtime = build_semantic_provider_runtime(settings, ontology=ontology)
    prototype_registry = PrototypeRegistry(
        ontology,
        embedding_provider=runtime.embedding_provider,
    )
    semantic_enricher = SemanticInferenceEnricher(
        ontology_registry=ontology,
        prototype_registry=prototype_registry,
        taxonomy_mapping_provider=(
            runtime.taxonomy_mapping_provider if settings.intelligence.semantic_assist_enabled else None
        ),
        reranker_provider=(
            runtime.reranker_provider if settings.intelligence.semantic_hypothesis_rerank_enabled else None
        ),
        embedding_provider=runtime.embedding_provider,
        semantic_assist_enabled=settings.intelligence.semantic_assist_enabled,
        semantic_hypothesis_rerank_enabled=settings.intelligence.semantic_hypothesis_rerank_enabled,
        semantic_corroboration_enabled=False,
        parallel_workers=1,
        parallel_min_mentions=0,
        parallel_chunk_size=max(settings.intelligence.semantic_parallel_chunk_size, 1),
        parallel_settings=None,
        runtime_diagnostics_enabled=settings.intelligence.semantic_runtime_diagnostics_enabled,
        orchestration_cache_enabled=settings.intelligence.semantic_orchestration_cache_enabled,
    )
    _WORKER_STATE = _SemanticWorkerState(runtime=runtime, semantic_enricher=semantic_enricher)
    atexit.register(_close_worker_resources)


def _process_semantic_chunk(
    mentions: list[MentionRecord],
    contexts: list[MentionContext],
    explicit_targets_by_mention: dict[str, list[TargetReference]],
) -> SemanticHypothesisBatch:
    if _WORKER_STATE is None:
        raise RuntimeError("Semantic worker state was not initialized.")
    return _WORKER_STATE.semantic_enricher.collect_hypotheses_for_chunk(
        mentions=mentions,
        contexts=contexts,
        explicit_targets_by_mention=explicit_targets_by_mention,
    )


def parallel_collect_hypotheses(
    *,
    settings_payload: dict[str, object],
    ontology_payload: dict[str, object],
    mentions: list[MentionRecord],
    contexts: list[MentionContext],
    explicit_targets_by_mention: dict[str, list[TargetReference]],
    workers: int,
    chunk_size: int,
) -> SemanticHypothesisBatch:
    if workers <= 1 or len(mentions) <= 1:
        raise ValueError("Parallel semantic collection requires at least two workers and two mentions.")

    context_by_id = {context.mention_id: context for context in contexts}
    effective_chunk_size = max(chunk_size, math.ceil(len(mentions) / max(workers * 2, 1)))
    mention_chunks = [
        mentions[index : index + effective_chunk_size]
        for index in range(0, len(mentions), effective_chunk_size)
    ]
    context_chunks = [
        [context_by_id[mention.mention_id] for mention in chunk if mention.mention_id in context_by_id]
        for chunk in mention_chunks
    ]

    ordered_batches: list[tuple[int, SemanticHypothesisBatch]] = []
    spawn_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(mention_chunks)),
        mp_context=spawn_context,
        initializer=_init_worker,
        initargs=(settings_payload, ontology_payload),
    ) as executor:
        futures = [
            (
                index,
                executor.submit(
                    _process_semantic_chunk,
                    chunk,
                    context_chunks[index],
                    explicit_targets_by_mention,
                ),
            )
            for index, chunk in enumerate(mention_chunks)
        ]
        for index, future in futures:
            ordered_batches.append((index, future.result()))

    combined = SemanticHypothesisBatch()
    intelligence_settings = settings_payload.get("intelligence", {})
    collect_runtime_stats = bool(
        intelligence_settings.get("semantic_runtime_diagnostics_enabled")
        if isinstance(intelligence_settings, dict)
        else False
    )
    started_at = perf_counter() if collect_runtime_stats else None
    combined_runtime = (
        {
            "timings": {"semantic_parallel_merge": 0.0},
            "counters": {"semantic_parallel_batches": len(ordered_batches)},
            "cache": {},
        }
        if collect_runtime_stats
        else None
    )
    for _, batch in sorted(ordered_batches, key=lambda item: item[0]):
        combined.mention_sentiments.extend(batch.mention_sentiments)
        combined.target_sentiments.extend(batch.target_sentiments)
        combined.aspect_opinions.extend(batch.aspect_opinions)
        combined.issue_signals.extend(batch.issue_signals)
        if not collect_runtime_stats or combined_runtime is None:
            continue
        runtime_stats = batch.runtime_stats if isinstance(batch.runtime_stats, dict) else {}
        for timing_name, value in dict(runtime_stats.get("timings", {})).items():
            if isinstance(value, (int, float)):
                timings = combined_runtime.setdefault("timings", {})
                timings[timing_name] = round(float(timings.get(timing_name, 0.0)) + float(value), 6)
        for counter_name, value in dict(runtime_stats.get("counters", {})).items():
            if isinstance(value, int):
                counters = combined_runtime.setdefault("counters", {})
                counters[counter_name] = int(counters.get(counter_name, 0)) + value
        for cache_name, stats in dict(runtime_stats.get("cache", {})).items():
            if not isinstance(stats, dict):
                continue
            cache_runtime = combined_runtime.setdefault("cache", {}).setdefault(
                cache_name,
                {"hits": 0, "misses": 0, "stores": 0, "entries": 0},
            )
            for field_name in ("hits", "misses", "stores", "entries"):
                value = stats.get(field_name)
                if isinstance(value, int):
                    cache_runtime[field_name] = int(cache_runtime.get(field_name, 0)) + value
    if collect_runtime_stats and combined_runtime is not None and started_at is not None:
        combined_runtime["timings"]["semantic_parallel_merge"] = round(perf_counter() - started_at, 6)
        combined.runtime_stats = combined_runtime
    return combined
