from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class StorageLayout:
    bronze_raw_jsonl: object
    silver_mentions: object
    silver_dedup_clusters: object
    silver_author_quality: object
    silver_thread_edges: object
    silver_thread_contexts: object
    silver_thread_summaries: object
    gold_entities: object
    gold_entity_candidate_clusters: object
    gold_keywords: object
    gold_topics: object
    gold_topic_artifacts: object
    gold_sentiment: object
    gold_target_sentiment: object
    gold_stance: object
    gold_intents: object
    gold_aspects: object
    gold_issues: object
    gold_sources: object
    marts_dir: object
    metrics_file: object
    bi_reports_file: object
    run_manifest_file: object
    insights_file: object
    embedding_cache_dir: object
    vector_index_dir: object
    topic_artifacts_dir: object
    latest_topic_artifact_snapshot: object
    semantic_promoted_store: object
    topic_lineage_store: object

    @classmethod
    def from_settings(cls, settings):
        settings.ensure_directories()
        return cls(bronze_raw_jsonl=settings.bronze_dir / 'uap_records.jsonl', silver_mentions=settings.silver_dir / 'mentions.parquet', silver_dedup_clusters=settings.silver_dir / 'dedup_clusters.parquet', silver_author_quality=settings.silver_dir / 'author_quality.parquet', silver_thread_edges=settings.silver_dir / 'thread_edges.parquet', silver_thread_contexts=settings.silver_dir / 'mention_contexts.parquet', silver_thread_summaries=settings.silver_dir / 'thread_summaries.parquet', gold_entities=settings.gold_dir / 'entity_facts.parquet', gold_entity_candidate_clusters=settings.gold_dir / 'entity_candidate_clusters.parquet', gold_keywords=settings.gold_dir / 'keyword_facts.parquet', gold_topics=settings.gold_dir / 'topic_facts.parquet', gold_topic_artifacts=settings.gold_dir / 'topic_artifacts.parquet', gold_sentiment=settings.gold_dir / 'sentiment_facts.parquet', gold_target_sentiment=settings.gold_dir / 'target_sentiment_facts.parquet', gold_stance=settings.gold_dir / 'stance_facts.parquet', gold_intents=settings.gold_dir / 'intent_facts.parquet', gold_aspects=settings.gold_dir / 'aspect_facts.parquet', gold_issues=settings.gold_dir / 'issue_facts.parquet', gold_sources=settings.gold_dir / 'source_influence_facts.parquet', marts_dir=settings.gold_dir / 'marts', metrics_file=settings.reports_dir / 'metrics.json', bi_reports_file=settings.reports_dir / 'bi_reports.json', run_manifest_file=settings.reports_dir / 'run_manifest.json', insights_file=settings.insights_dir / 'insights.jsonl', embedding_cache_dir=settings.embedding_cache_dir, vector_index_dir=settings.vector_index_dir, topic_artifacts_dir=settings.topic_artifacts_dir, latest_topic_artifact_snapshot=settings.latest_topic_artifact_snapshot_path, semantic_promoted_store=settings.semantic_promoted_path, topic_lineage_store=settings.topic_lineage_path)
