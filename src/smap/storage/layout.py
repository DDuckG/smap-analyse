from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smap.core.settings import Settings


@dataclass(frozen=True, slots=True)
class StorageLayout:
    bronze_raw_jsonl: Path
    silver_mentions: Path
    silver_dedup_clusters: Path
    silver_author_quality: Path
    silver_thread_edges: Path
    silver_thread_contexts: Path
    silver_thread_summaries: Path
    gold_entities: Path
    gold_entity_candidate_clusters: Path
    gold_keywords: Path
    gold_topics: Path
    gold_topic_artifacts: Path
    gold_sentiment: Path
    gold_target_sentiment: Path
    gold_stance: Path
    gold_intents: Path
    gold_aspects: Path
    gold_issues: Path
    gold_sources: Path
    marts_dir: Path
    metrics_file: Path
    bi_reports_file: Path
    run_manifest_file: Path
    qa_reports_file: Path
    qa_summary_file: Path
    regression_summary_file: Path
    release_summary_file: Path
    insights_file: Path
    exports_dir: Path
    viewer_dir: Path
    viewer_index_file: Path
    staging_dir: Path
    sample_input_manifest_file: Path
    embedding_cache_dir: Path
    vector_index_dir: Path
    topic_artifacts_dir: Path
    latest_topic_artifact_snapshot: Path
    semantic_feedback_store: Path
    semantic_promoted_store: Path
    semantic_benchmark_gold_store: Path
    topic_feedback_store: Path
    topic_lineage_store: Path
    label_studio_export_dir: Path
    label_studio_import_dir: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> StorageLayout:
        settings.ensure_directories()
        return cls(
            bronze_raw_jsonl=settings.bronze_dir / "uap_records.jsonl",
            silver_mentions=settings.silver_dir / "mentions.parquet",
            silver_dedup_clusters=settings.silver_dir / "dedup_clusters.parquet",
            silver_author_quality=settings.silver_dir / "author_quality.parquet",
            silver_thread_edges=settings.silver_dir / "thread_edges.parquet",
            silver_thread_contexts=settings.silver_dir / "mention_contexts.parquet",
            silver_thread_summaries=settings.silver_dir / "thread_summaries.parquet",
            gold_entities=settings.gold_dir / "entity_facts.parquet",
            gold_entity_candidate_clusters=settings.gold_dir / "entity_candidate_clusters.parquet",
            gold_keywords=settings.gold_dir / "keyword_facts.parquet",
            gold_topics=settings.gold_dir / "topic_facts.parquet",
            gold_topic_artifacts=settings.gold_dir / "topic_artifacts.parquet",
            gold_sentiment=settings.gold_dir / "sentiment_facts.parquet",
            gold_target_sentiment=settings.gold_dir / "target_sentiment_facts.parquet",
            gold_stance=settings.gold_dir / "stance_facts.parquet",
            gold_intents=settings.gold_dir / "intent_facts.parquet",
            gold_aspects=settings.gold_dir / "aspect_facts.parquet",
            gold_issues=settings.gold_dir / "issue_facts.parquet",
            gold_sources=settings.gold_dir / "source_influence_facts.parquet",
            marts_dir=settings.gold_dir / "marts",
            metrics_file=settings.reports_dir / "metrics.json",
            bi_reports_file=settings.reports_dir / "bi_reports.json",
            run_manifest_file=settings.reports_dir / "run_manifest.json",
            qa_reports_file=settings.reports_dir / "qa_reports.json",
            qa_summary_file=settings.reports_dir / "qa_summary.json",
            regression_summary_file=settings.reports_dir / "regression_summary.json",
            release_summary_file=settings.reports_dir / "release_summary.json",
            insights_file=settings.insights_dir / "insights.jsonl",
            exports_dir=settings.reports_dir / "exports",
            viewer_dir=settings.reports_dir / "viewer",
            viewer_index_file=settings.reports_dir / "viewer" / "index.html",
            staging_dir=settings.data_dir / "staging",
            sample_input_manifest_file=settings.reports_dir / "sample_input_manifest.json",
            embedding_cache_dir=settings.embedding_cache_dir,
            vector_index_dir=settings.vector_index_dir,
            topic_artifacts_dir=settings.topic_artifacts_dir,
            latest_topic_artifact_snapshot=settings.latest_topic_artifact_snapshot_path,
            semantic_feedback_store=settings.semantic_feedback_path,
            semantic_promoted_store=settings.semantic_promoted_path,
            semantic_benchmark_gold_store=settings.semantic_benchmark_gold_path,
            topic_feedback_store=settings.topic_feedback_path,
            topic_lineage_store=settings.topic_lineage_path,
            label_studio_export_dir=settings.label_studio_export_dir,
            label_studio_import_dir=settings.label_studio_import_dir,
        )
