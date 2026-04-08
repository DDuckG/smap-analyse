from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, Field

from smap.analytics.metrics import build_metrics
from smap.bi.models import BIReportBundle
from smap.bi.reports import build_bi_reports
from smap.canonicalization.alias import AliasRegistry
from smap.canonicalization.service import CanonicalizationEngine
from smap.core.exceptions import PipelineError
from smap.core.settings import Settings, get_settings
from smap.dedup.models import DedupClusterRecord
from smap.dedup.service import DeduplicationService
from smap.enrichers.entity import EntityExtractionEnricher
from smap.enrichers.models import EnrichmentBundle
from smap.enrichers.semantic import SemanticInferenceEnricher
from smap.enrichers.service import EnricherService
from smap.enrichers.topic import TopicCandidateEnricher
from smap.enrichers.topic_quality import persist_topic_artifact_snapshot
from smap.hil.feedback_store import FeedbackStore
from smap.ingestion.models import IngestedBatchBundle
from smap.ingestion.service import ingest_batch
from smap.insights.generator import generate_insights
from smap.insights.models import InsightCard
from smap.marts.builder import MartBundle, build_marts
from smap.normalization.models import MentionRecord
from smap.normalization.service import normalize_batch
from smap.ontology.alignment import collect_alignment_errors
from smap.ontology.models import OntologyRegistry
from smap.ontology.prototypes import PrototypeRegistry
from smap.ontology.runtime import OntologyRuntime, load_runtime_ontology
from smap.providers.factory import (
    ProviderRuntime,
    build_ner_providers_with_embedding,
    build_provider_runtime,
)
from smap.quality.models import AuthorQualityRecord
from smap.quality.spam import SpamScoringService
from smap.review.context import build_review_context_index
from smap.review.db import session_scope
from smap.review.metrics import ReviewQueueSummary
from smap.review.migrations import require_database_ready
from smap.review.service import queue_review_items, seed_ontology_version
from smap.run_manifest import RunManifest, build_run_manifest, default_run_id
from smap.runtime.timing import StageProgressCallback, StageTimingCollector
from smap.storage.layout import StorageLayout
from smap.storage.repository import (
    copy_file_if_needed,
    register_parquet_tables,
    write_models_parquet,
    write_table_bundle,
    write_text_if_changed,
)
from smap.threads.models import ThreadBundle
from smap.threads.service import build_threads
from smap.validation.models import BatchProfile, ValidationReport


class PipelineRunResult(BaseModel):
    run_id: str
    input_path: str
    total_valid_records: int
    validation_report: ValidationReport
    batch_profile: BatchProfile
    metrics: dict[str, object]
    bi_reports: BIReportBundle | None = None
    run_manifest: RunManifest | None = None
    storage_paths: dict[str, str] = Field(default_factory=dict)
    review_items_created: int = 0
    review_queue_summary: ReviewQueueSummary = Field(default_factory=ReviewQueueSummary)
    insight_cards: list[InsightCard] = Field(default_factory=list)
    stage_timings: dict[str, float] = Field(default_factory=dict)


def _ingested_bundle(
    input_source: Path | IngestedBatchBundle,
    *,
    materialized_raw_jsonl_path: Path | None = None,
) -> IngestedBatchBundle:
    if isinstance(input_source, IngestedBatchBundle):
        return input_source
    return ingest_batch(input_source, materialized_raw_jsonl_path=materialized_raw_jsonl_path)


def _build_enricher_service(
    settings: Settings,
    ontology: OntologyRegistry,
    *,
    runtime: ProviderRuntime | None = None,
    other_domain_aliases: frozenset[str] | None = None,
) -> tuple[EnricherService, ProviderRuntime]:
    runtime = runtime or build_provider_runtime(settings, ontology=ontology)
    prototype_registry = PrototypeRegistry(
        ontology,
        embedding_provider=runtime.embedding_provider,
    )
    set_phrase_lexicon = getattr(runtime.embedding_provider, "set_phrase_lexicon", None)
    if callable(set_phrase_lexicon):
        set_phrase_lexicon(prototype_registry.phrase_lexicon)
    alias_registry = AliasRegistry.from_ontology(ontology)
    feedback_store = FeedbackStore(settings)
    entity_engine = CanonicalizationEngine(
        alias_registry=alias_registry,
        embedding_provider=runtime.embedding_provider,
        prototype_registry=prototype_registry,
        vector_index=runtime.vector_index,
        embedding_rerank_enabled=settings.intelligence.entity_embedding_rerank_enabled,
        verification_level=settings.intelligence.verifiers.entity_level,
    )
    entity_enricher = EntityExtractionEnricher(
        entity_engine,
        ontology_registry=ontology,
        other_domain_aliases=other_domain_aliases or frozenset(),
        ner_provider_builder=lambda scoped_alias_registry: build_ner_providers_with_embedding(
            settings,
            scoped_alias_registry,
            embedding_provider=runtime.embedding_provider,
        ),
    )
    topic_enricher = TopicCandidateEnricher(
        runtime.topic_provider,
        artifact_history_dir=settings.topic_artifacts_dir,
        reviewed_topic_lineage=feedback_store.load_topic_lineage(),
        embedding_provider=runtime.embedding_provider,
        prototype_registry=prototype_registry,
        artifact_level=settings.intelligence.verifiers.topic_artifact_level,
        lineage_level=settings.intelligence.verifiers.lineage_level,
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
        promoted_semantic_knowledge=feedback_store.load_promoted_semantic_knowledge(),
        semantic_assist_enabled=settings.intelligence.semantic_assist_enabled,
        semantic_hypothesis_rerank_enabled=settings.intelligence.semantic_hypothesis_rerank_enabled,
        semantic_corroboration_enabled=settings.intelligence.semantic_corroboration_enabled,
        parallel_workers=settings.intelligence.semantic_parallel_workers,
        parallel_min_mentions=settings.intelligence.semantic_parallel_min_mentions,
        parallel_chunk_size=settings.intelligence.semantic_parallel_chunk_size,
        parallel_settings=settings,
        runtime_diagnostics_enabled=settings.intelligence.semantic_runtime_diagnostics_enabled,
        orchestration_cache_enabled=settings.intelligence.semantic_orchestration_cache_enabled,
    )
    return (
        EnricherService(
            entity_enricher=entity_enricher,
            topic_enricher=topic_enricher,
            semantic_enricher=semantic_enricher,
        ),
        runtime,
    )


def _close_runtime_resources(runtime: ProviderRuntime) -> None:
    seen: set[int] = set()
    resources = [
        runtime.language_id_provider,
        runtime.embedding_provider,
        runtime.vector_index,
        runtime.topic_provider,
        runtime.taxonomy_mapping_provider,
        runtime.reranker_provider,
        getattr(runtime.embedding_provider, "cache_store", None),
        getattr(getattr(runtime.taxonomy_mapping_provider, "embedding_provider", None), "cache_store", None),
        getattr(getattr(runtime.reranker_provider, "embedding_provider", None), "cache_store", None),
    ]
    for resource in resources:
        if resource is None:
            continue
        resource_id = id(resource)
        close_method = getattr(resource, "close", None)
        if resource_id in seen or not callable(close_method):
            continue
        seen.add(resource_id)
        close_method()


def run_pipeline(
    input_path: Path | IngestedBatchBundle,
    settings: Settings | None = None,
    *,
    run_id: str | None = None,
    progress_callback: StageProgressCallback | None = None,
) -> PipelineRunResult:
    settings = settings or get_settings()
    settings.ensure_directories()
    layout = StorageLayout.from_settings(settings)
    pipeline_run_id = run_id or default_run_id()
    timings = StageTimingCollector(progress_callback=progress_callback)
    provider_runtime: ProviderRuntime | None = None
    ontology_runtime: OntologyRuntime | None = None
    dedup_clusters: list[DedupClusterRecord] = []
    author_quality_profiles: list[AuthorQualityRecord] = []

    try:
        with timings.stage("ingest"):
            ingested = _ingested_bundle(input_path, materialized_raw_jsonl_path=layout.bronze_raw_jsonl)
        with timings.stage("ontology"):
            ontology_runtime = load_runtime_ontology(settings, records=ingested.parsed_records)
            ontology = ontology_runtime.registry
        provider_runtime = build_provider_runtime(settings, ontology=ontology)
        with timings.stage("normalize"):
            normalized = normalize_batch(
                ingested.parsed_records,
                language_id_provider=provider_runtime.language_id_provider,
            )
        if settings.analytics.dedup.enabled:
            with timings.stage("dedup"):
                dedup_result = DeduplicationService(
                    exact_enabled=settings.analytics.dedup.exact_enabled,
                    near_enabled=settings.analytics.dedup.near_enabled,
                    min_text_length=settings.analytics.dedup.min_text_length,
                    word_shingle_size=settings.analytics.dedup.word_shingle_size,
                    char_shingle_size=settings.analytics.dedup.char_shingle_size,
                    num_perm=settings.analytics.dedup.num_perm,
                    num_bands=settings.analytics.dedup.num_bands,
                    near_similarity_threshold=settings.analytics.dedup.near_similarity_threshold,
                    max_bucket_size=settings.analytics.dedup.max_bucket_size,
                ).annotate(normalized.mentions)
                normalized = normalized.model_copy(update={"mentions": dedup_result[0]})
                dedup_clusters = dedup_result[1].clusters
        if settings.analytics.spam.enabled:
            with timings.stage("spam"):
                quality_result = SpamScoringService(
                    mention_threshold=settings.analytics.spam.mention_threshold,
                    author_threshold=settings.analytics.spam.author_threshold,
                    burst_window_minutes=settings.analytics.spam.burst_window_minutes,
                    quality_weight_floor=settings.analytics.spam.quality_weight_floor,
                    mention_discount_strength=settings.analytics.spam.mention_discount_strength,
                    author_discount_strength=settings.analytics.spam.author_discount_strength,
                ).annotate(normalized.mentions)
                normalized = normalized.model_copy(update={"mentions": quality_result[0]})
                author_quality_profiles = quality_result[1].author_profiles
        with timings.stage("threads"):
            threads = build_threads(normalized.mentions)
        enricher_service, provider_runtime = _build_enricher_service(
            settings,
            ontology,
            runtime=provider_runtime,
            other_domain_aliases=(
                ontology_runtime.other_domain_aliases if ontology_runtime is not None else frozenset()
            ),
        )
        context_map = {context.mention_id: context for context in threads.contexts}
        enrichment = EnrichmentBundle()
        with timings.stage("entity"):
            enricher_service.entity_enricher.prepare(normalized.mentions, threads.contexts)
            for mention in normalized.mentions:
                context = context_map.get(mention.mention_id)
                enrichment.entity_facts.extend(enricher_service.entity_enricher.enrich(mention, context))
                enrichment.keyword_facts.extend(enricher_service.keyword_enricher.enrich(mention, context))
                enrichment.stance_facts.extend(enricher_service.stance_enricher.enrich(mention, context))
                enrichment.intent_facts.extend(enricher_service.intent_enricher.enrich(mention, context))
                enrichment.source_influence_facts.extend(enricher_service.source_enricher.enrich(mention, context))
            (
                enrichment.entity_facts,
                enrichment.entity_candidate_clusters,
            ) = enricher_service.entity_enricher.annotate_batch_local_candidates(
                enrichment.entity_facts,
                normalized.mentions,
            )
        with timings.stage("semantic"):
            semantic_result = enricher_service.semantic_enricher.enrich(
                normalized.mentions,
                threads.contexts,
                enrichment.entity_facts,
            )
            enrichment.sentiment_facts.extend(semantic_result.mention_sentiments)
            enrichment.target_sentiment_facts.extend(semantic_result.target_sentiments)
            enrichment.aspect_opinion_facts.extend(semantic_result.aspect_opinions)
            enrichment.issue_signal_facts.extend(semantic_result.issue_signals)
            alignment_errors = collect_alignment_errors(enrichment, ontology)
            if alignment_errors:
                raise PipelineError(f"Enrichment labels do not align with ontology: {alignment_errors}")
        with timings.stage("topic"):
            enricher_service.topic_enricher.prepare(
                normalized.mentions,
                threads.contexts,
                entity_facts=enrichment.entity_facts,
                aspect_facts=enrichment.aspect_opinion_facts,
                issue_facts=enrichment.issue_signal_facts,
            )
            for mention in normalized.mentions:
                context = context_map.get(mention.mention_id)
                enrichment.topic_facts.extend(enricher_service.topic_enricher.enrich(mention, context))
            enrichment.topic_artifacts.extend(enricher_service.topic_enricher.artifacts())

        require_database_ready(settings)
        review_items_created = 0
        review_queue_summary = ReviewQueueSummary()
        with timings.stage("review"):
            review_contexts = build_review_context_index(normalized.mentions, ontology)
            with session_scope(settings) as session:
                seed_ontology_version(session, ontology)
                review_queue_summary = queue_review_items(
                    session,
                    enrichment,
                    settings=settings,
                    review_contexts=review_contexts,
                )
                review_items_created = review_queue_summary.created_items

        with timings.stage("persist_facts"):
            if ingested.raw_jsonl_path is not None:
                copy_file_if_needed(ingested.raw_jsonl_path, layout.bronze_raw_jsonl)
            elif ingested.path.is_file() and ingested.path.suffix == ".jsonl":
                copy_file_if_needed(ingested.path, layout.bronze_raw_jsonl)
            elif not layout.bronze_raw_jsonl.exists():
                # Bundle callers normally pass a materialized raw jsonl path. This reread is a compatibility fallback.
                ingest_batch(ingested.path, materialized_raw_jsonl_path=layout.bronze_raw_jsonl)
            write_models_parquet(layout.silver_mentions, normalized.mentions)
            write_models_parquet(layout.silver_dedup_clusters, dedup_clusters)
            write_models_parquet(layout.silver_author_quality, author_quality_profiles)
            write_models_parquet(layout.silver_thread_edges, threads.edges)
            write_models_parquet(layout.silver_thread_contexts, threads.contexts)
            write_models_parquet(layout.silver_thread_summaries, threads.summaries)
            write_models_parquet(layout.gold_entities, enrichment.entity_facts)
            write_models_parquet(layout.gold_entity_candidate_clusters, enrichment.entity_candidate_clusters)
            write_models_parquet(layout.gold_keywords, enrichment.keyword_facts)
            write_models_parquet(layout.gold_topics, enrichment.topic_facts)
            write_models_parquet(layout.gold_topic_artifacts, enrichment.topic_artifacts)
            topic_artifact_snapshot_path = persist_topic_artifact_snapshot(layout.topic_artifacts_dir, enrichment.topic_artifacts)
            write_models_parquet(layout.gold_sentiment, enrichment.sentiment_facts)
            write_models_parquet(layout.gold_target_sentiment, enrichment.target_sentiment_facts)
            write_models_parquet(layout.gold_stance, enrichment.stance_facts)
            write_models_parquet(layout.gold_intents, enrichment.intent_facts)
            write_models_parquet(layout.gold_aspects, enrichment.aspect_opinion_facts)
            write_models_parquet(layout.gold_issues, enrichment.issue_signal_facts)
            write_models_parquet(layout.gold_sources, enrichment.source_influence_facts)

        with timings.stage("marts"):
            marts = build_marts(
                normalized.mentions,
                threads,
                enrichment,
                ontology,
                dedup_clusters=dedup_clusters,
                author_quality=author_quality_profiles,
            )
            table_paths = write_table_bundle(layout.marts_dir, marts.tables)
            register_parquet_tables(settings.analytics_duckdb, table_paths)
        with timings.stage("metrics"):
            metrics = build_metrics(marts, weighting_mode=settings.analytics.weighting_mode)
        with timings.stage("bi_reports"):
            bi_reports = build_bi_reports(
                marts,
                metrics=metrics,
                weighting_mode=settings.analytics.weighting_mode,
            )
        with timings.stage("insights"):
            insights = generate_insights(bi_reports)

        with timings.stage("persist_reports"):
            layout.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            write_text_if_changed(layout.metrics_file, json.dumps(metrics, ensure_ascii=False, indent=2))
            write_text_if_changed(layout.bi_reports_file, bi_reports.model_dump_json(indent=2))
            write_text_if_changed(
                layout.insights_file,
                "\n".join(card.model_dump_json() for card in insights),
            )

        storage_paths = {
            "bronze_raw_jsonl": str(layout.bronze_raw_jsonl),
            "silver_mentions": str(layout.silver_mentions),
            "silver_dedup_clusters": str(layout.silver_dedup_clusters),
            "silver_author_quality": str(layout.silver_author_quality),
            "silver_thread_edges": str(layout.silver_thread_edges),
            "silver_thread_contexts": str(layout.silver_thread_contexts),
            "silver_thread_summaries": str(layout.silver_thread_summaries),
            "gold_entities": str(layout.gold_entities),
            "gold_entity_candidate_clusters": str(layout.gold_entity_candidate_clusters),
            "gold_keywords": str(layout.gold_keywords),
            "gold_topics": str(layout.gold_topics),
            "gold_topic_artifacts": str(layout.gold_topic_artifacts),
            "topic_artifact_snapshot": str(topic_artifact_snapshot_path),
            "gold_sentiment": str(layout.gold_sentiment),
            "gold_target_sentiment": str(layout.gold_target_sentiment),
            "gold_stance": str(layout.gold_stance),
            "gold_intents": str(layout.gold_intents),
            "gold_aspects": str(layout.gold_aspects),
            "gold_issues": str(layout.gold_issues),
            "gold_sources": str(layout.gold_sources),
            "metrics": str(layout.metrics_file),
            "bi_reports": str(layout.bi_reports_file),
            "run_manifest": str(layout.run_manifest_file),
            "insights": str(layout.insights_file),
            "vector_index_dir": str(layout.vector_index_dir),
            "semantic_feedback_store": str(layout.semantic_feedback_store),
            "semantic_promoted_store": str(layout.semantic_promoted_store),
            "semantic_benchmark_gold_store": str(layout.semantic_benchmark_gold_store),
            "topic_feedback_store": str(layout.topic_feedback_store),
            "topic_lineage_store": str(layout.topic_lineage_store),
        }
        storage_paths.update({table_name: str(path) for table_name, path in table_paths.items()})

        with timings.stage("manifest"):
            run_manifest_model = build_run_manifest(
                run_id=pipeline_run_id,
                input_path=ingested.path,
                settings=settings,
                enrichment=enrichment,
                bi_reports=bi_reports,
                storage_paths=storage_paths,
                ontology_runtime=ontology_runtime,
                input_inspection=ingested.input_inspection,
                record_counts={
                    "total_records": ingested.validation_report.total_records,
                    "valid_records": ingested.validation_report.valid_records,
                    "invalid_records": ingested.validation_report.invalid_records,
                },
                provider_runtime=provider_runtime,
                stage_timings=timings.stage_seconds,
            )
            layout.run_manifest_file.parent.mkdir(parents=True, exist_ok=True)
            write_text_if_changed(layout.run_manifest_file, run_manifest_model.model_dump_json(indent=2))
        run_manifest_model = run_manifest_model.model_copy(update={"stage_timings": timings.stage_seconds})
        write_text_if_changed(layout.run_manifest_file, run_manifest_model.model_dump_json(indent=2))

        return PipelineRunResult(
            run_id=pipeline_run_id,
            input_path=ingested.input_path,
            total_valid_records=len(ingested.parsed_records),
            validation_report=ingested.validation_report,
            batch_profile=ingested.batch_profile,
            metrics=metrics,
            bi_reports=bi_reports,
            run_manifest=run_manifest_model,
            storage_paths=storage_paths,
            review_items_created=review_items_created,
            review_queue_summary=review_queue_summary,
            insight_cards=insights,
            stage_timings=timings.stage_seconds,
        )
    finally:
        if provider_runtime is not None:
            _close_runtime_resources(provider_runtime)


def normalize_to_mentions(input_path: Path | IngestedBatchBundle) -> list[MentionRecord]:
    ingested = _ingested_bundle(input_path)
    runtime = build_provider_runtime(get_settings())
    try:
        return normalize_batch(
            ingested.parsed_records,
            language_id_provider=runtime.language_id_provider,
        ).mentions
    finally:
        _close_runtime_resources(runtime)


def build_threads_for_input(input_path: Path | IngestedBatchBundle) -> ThreadBundle:
    mentions = normalize_to_mentions(input_path)
    return build_threads(mentions)


def build_marts_for_input(
    input_path: Path | IngestedBatchBundle,
    settings: Settings | None = None,
) -> MartBundle:
    settings = settings or get_settings()
    ingested = _ingested_bundle(input_path)
    ontology_runtime = load_runtime_ontology(settings, records=ingested.parsed_records)
    ontology = ontology_runtime.registry
    runtime = build_provider_runtime(settings, ontology=ontology)
    normalized = normalize_batch(
        ingested.parsed_records,
        language_id_provider=runtime.language_id_provider,
    )
    dedup_clusters: list[DedupClusterRecord] = []
    if settings.analytics.dedup.enabled:
        normalized_mentions, dedup_result = DeduplicationService(
            exact_enabled=settings.analytics.dedup.exact_enabled,
            near_enabled=settings.analytics.dedup.near_enabled,
            min_text_length=settings.analytics.dedup.min_text_length,
            word_shingle_size=settings.analytics.dedup.word_shingle_size,
            char_shingle_size=settings.analytics.dedup.char_shingle_size,
            num_perm=settings.analytics.dedup.num_perm,
            num_bands=settings.analytics.dedup.num_bands,
            near_similarity_threshold=settings.analytics.dedup.near_similarity_threshold,
            max_bucket_size=settings.analytics.dedup.max_bucket_size,
        ).annotate(normalized.mentions)
        normalized = normalized.model_copy(update={"mentions": normalized_mentions})
        dedup_clusters = dedup_result.clusters
    author_quality_profiles: list[AuthorQualityRecord] = []
    if settings.analytics.spam.enabled:
        normalized_mentions, quality_result = SpamScoringService(
            mention_threshold=settings.analytics.spam.mention_threshold,
            author_threshold=settings.analytics.spam.author_threshold,
            burst_window_minutes=settings.analytics.spam.burst_window_minutes,
            quality_weight_floor=settings.analytics.spam.quality_weight_floor,
            mention_discount_strength=settings.analytics.spam.mention_discount_strength,
            author_discount_strength=settings.analytics.spam.author_discount_strength,
        ).annotate(normalized.mentions)
        normalized = normalized.model_copy(update={"mentions": normalized_mentions})
        author_quality_profiles = quality_result.author_profiles
    threads = build_threads(normalized.mentions)
    enrichment_service, runtime = _build_enricher_service(
        settings,
        ontology,
        runtime=runtime,
        other_domain_aliases=ontology_runtime.other_domain_aliases,
    )
    try:
        enrichment = enrichment_service.enrich_mentions(
            normalized.mentions,
            threads.contexts,
        )
        alignment_errors = collect_alignment_errors(enrichment, ontology)
        if alignment_errors:
            raise PipelineError(f"Enrichment labels do not align with ontology: {alignment_errors}")
        return build_marts(
            normalized.mentions,
            threads,
            enrichment,
            ontology,
            dedup_clusters=dedup_clusters,
            author_quality=author_quality_profiles,
        )
    finally:
        _close_runtime_resources(runtime)


def build_enrichment_for_input(
    input_path: Path | IngestedBatchBundle,
    settings: Settings | None = None,
) -> tuple[list[MentionRecord], ThreadBundle, EnrichmentBundle]:
    settings = settings or get_settings()
    ingested = _ingested_bundle(input_path)
    ontology_runtime = load_runtime_ontology(settings, records=ingested.parsed_records)
    ontology = ontology_runtime.registry
    runtime = build_provider_runtime(settings, ontology=ontology)
    normalized = normalize_batch(
        ingested.parsed_records,
        language_id_provider=runtime.language_id_provider,
    )
    if settings.analytics.dedup.enabled:
        normalized_mentions, _ = DeduplicationService(
            exact_enabled=settings.analytics.dedup.exact_enabled,
            near_enabled=settings.analytics.dedup.near_enabled,
            min_text_length=settings.analytics.dedup.min_text_length,
            word_shingle_size=settings.analytics.dedup.word_shingle_size,
            char_shingle_size=settings.analytics.dedup.char_shingle_size,
            num_perm=settings.analytics.dedup.num_perm,
            num_bands=settings.analytics.dedup.num_bands,
            near_similarity_threshold=settings.analytics.dedup.near_similarity_threshold,
            max_bucket_size=settings.analytics.dedup.max_bucket_size,
        ).annotate(normalized.mentions)
        normalized = normalized.model_copy(update={"mentions": normalized_mentions})
    if settings.analytics.spam.enabled:
        normalized_mentions, _ = SpamScoringService(
            mention_threshold=settings.analytics.spam.mention_threshold,
            author_threshold=settings.analytics.spam.author_threshold,
            burst_window_minutes=settings.analytics.spam.burst_window_minutes,
            quality_weight_floor=settings.analytics.spam.quality_weight_floor,
            mention_discount_strength=settings.analytics.spam.mention_discount_strength,
            author_discount_strength=settings.analytics.spam.author_discount_strength,
        ).annotate(normalized.mentions)
        normalized = normalized.model_copy(update={"mentions": normalized_mentions})
    threads = build_threads(normalized.mentions)
    enrichment_service, runtime = _build_enricher_service(
        settings,
        ontology,
        runtime=runtime,
        other_domain_aliases=ontology_runtime.other_domain_aliases,
    )
    try:
        enrichment = enrichment_service.enrich_mentions(
            normalized.mentions,
            threads.contexts,
        )
        alignment_errors = collect_alignment_errors(enrichment, ontology)
        if alignment_errors:
            raise PipelineError(f"Enrichment labels do not align with ontology: {alignment_errors}")
        return normalized.mentions, threads, enrichment
    finally:
        _close_runtime_resources(runtime)


def inspect_semantic_runtime(
    input_path: Path | IngestedBatchBundle,
    settings: Settings | None = None,
) -> dict[str, object]:
    settings = settings or get_settings()
    inspect_settings = settings.model_copy(
        update={
            "intelligence": settings.intelligence.model_copy(
                update={
                    "semantic_runtime_diagnostics_enabled": True,
                    "semantic_orchestration_cache_enabled": True,
                }
            )
        }
    )
    ingested = _ingested_bundle(input_path)
    ontology_runtime = load_runtime_ontology(inspect_settings, records=ingested.parsed_records)
    ontology = ontology_runtime.registry
    runtime = build_provider_runtime(inspect_settings, ontology=ontology)
    normalized = normalize_batch(
        ingested.parsed_records,
        language_id_provider=runtime.language_id_provider,
    )
    if inspect_settings.analytics.dedup.enabled:
        normalized_mentions, _ = DeduplicationService(
            exact_enabled=inspect_settings.analytics.dedup.exact_enabled,
            near_enabled=inspect_settings.analytics.dedup.near_enabled,
            min_text_length=inspect_settings.analytics.dedup.min_text_length,
            word_shingle_size=inspect_settings.analytics.dedup.word_shingle_size,
            char_shingle_size=inspect_settings.analytics.dedup.char_shingle_size,
            num_perm=inspect_settings.analytics.dedup.num_perm,
            num_bands=inspect_settings.analytics.dedup.num_bands,
            near_similarity_threshold=inspect_settings.analytics.dedup.near_similarity_threshold,
            max_bucket_size=inspect_settings.analytics.dedup.max_bucket_size,
        ).annotate(normalized.mentions)
        normalized = normalized.model_copy(update={"mentions": normalized_mentions})
    if inspect_settings.analytics.spam.enabled:
        normalized_mentions, _ = SpamScoringService(
            mention_threshold=inspect_settings.analytics.spam.mention_threshold,
            author_threshold=inspect_settings.analytics.spam.author_threshold,
            burst_window_minutes=inspect_settings.analytics.spam.burst_window_minutes,
            quality_weight_floor=inspect_settings.analytics.spam.quality_weight_floor,
            mention_discount_strength=inspect_settings.analytics.spam.mention_discount_strength,
            author_discount_strength=inspect_settings.analytics.spam.author_discount_strength,
        ).annotate(normalized.mentions)
        normalized = normalized.model_copy(update={"mentions": normalized_mentions})
    threads = build_threads(normalized.mentions)
    enrichment_service, runtime = _build_enricher_service(
        inspect_settings,
        ontology,
        runtime=runtime,
        other_domain_aliases=ontology_runtime.other_domain_aliases,
    )
    try:
        started_at = perf_counter()
        enrichment = enrichment_service.enrich_mentions(normalized.mentions, threads.contexts)
        elapsed_seconds = perf_counter() - started_at
        semantic_enricher = enrichment_service.semantic_enricher
        runtime_snapshot = getattr(runtime.embedding_provider, "runtime_snapshot", None)
        embedding_runtime = (
            runtime_snapshot()
            if callable(runtime_snapshot)
            else dict(getattr(runtime.embedding_provider.provenance, "run_metadata", {}))
        )
        return {
            "input_path": str(input_path),
            "mention_count": len(normalized.mentions),
            "semantic_parallel_workers": inspect_settings.intelligence.semantic_parallel_workers,
            "semantic_parallel_min_mentions": inspect_settings.intelligence.semantic_parallel_min_mentions,
            "embedding_backend": inspect_settings.intelligence.embeddings.runtime_backend,
            "embedding_batch_size": inspect_settings.intelligence.embeddings.batch_size,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "output_counts": {
                "entity_facts": len(enrichment.entity_facts),
                "topic_facts": len(enrichment.topic_facts),
                "sentiment_facts": len(enrichment.sentiment_facts),
                "aspect_opinion_facts": len(enrichment.aspect_opinion_facts),
                "issue_signal_facts": len(enrichment.issue_signal_facts),
            },
            "embedding_runtime": embedding_runtime,
            "semantic_cache": semantic_enricher.cache_stats_snapshot(),
            "semantic_runtime": semantic_enricher.semantic_runtime_snapshot(),
        }
    finally:
        _close_runtime_resources(runtime)


def load_pipeline_result_from_artifacts(settings: Settings | None = None) -> PipelineRunResult | None:
    settings = settings or get_settings()
    layout = StorageLayout.from_settings(settings)
    if not layout.run_manifest_file.exists() or not layout.bi_reports_file.exists():
        return None
    run_manifest = RunManifest.model_validate_json(layout.run_manifest_file.read_text(encoding="utf-8"))
    bi_reports = BIReportBundle.model_validate_json(layout.bi_reports_file.read_text(encoding="utf-8"))
    metrics = (
        json.loads(layout.metrics_file.read_text(encoding="utf-8"))
        if layout.metrics_file.exists()
        else {}
    )
    insight_cards: list[InsightCard] = []
    if layout.insights_file.exists():
        for line in layout.insights_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                insight_cards.append(InsightCard.model_validate_json(line))
    actual_input_path = (
        run_manifest.sample_input_selection.staged_input_path
        if run_manifest.sample_input_selection is not None
        else run_manifest.input_inspection.input_path
    )
    storage_paths = {
        artifact.artifact_name: artifact.path
        for artifact in run_manifest.output_artifacts
    }
    valid_records = int(run_manifest.record_counts.get("valid_records", 0))
    total_records = int(run_manifest.record_counts.get("total_records", valid_records))
    invalid_records = int(run_manifest.record_counts.get("invalid_records", max(total_records - valid_records, 0)))
    return PipelineRunResult(
        run_id=run_manifest.run_id,
        input_path=actual_input_path,
        total_valid_records=valid_records,
        validation_report=ValidationReport(
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
        ),
        batch_profile=BatchProfile(
            total_records=total_records,
            records_by_type={},
            records_by_platform={},
            field_profiles=[],
            text_length={},
            notes=["Loaded from persisted pipeline artifacts."],
        ),
        metrics=metrics,
        bi_reports=bi_reports,
        run_manifest=run_manifest,
        storage_paths=storage_paths,
        review_items_created=0,
        review_queue_summary=ReviewQueueSummary(),
        insight_cards=insight_cards,
        stage_timings=run_manifest.stage_timings,
    )
