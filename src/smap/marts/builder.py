from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import polars as pl
from pydantic import BaseModel
from smap.dedup.models import DedupClusterRecord
from smap.enrichers.models import AspectOpinionFact, EnrichmentBundle, EntityCandidateClusterFact, EntityFact, IntentFact, IssueSignalFact, SentimentFact, TargetSentimentFact, TopicArtifactFact, TopicFact
from smap.normalization.models import MentionRecord
from smap.ontology.models import OntologyRegistry
from smap.quality.models import AuthorQualityRecord
from smap.threads.models import ThreadBundle, ThreadSummary

@dataclass(slots=True)
class MartBundle:
    tables: dict[str, pl.DataFrame]

def _frame_from_models(records, model_cls):
    payload = [record.model_dump(mode='json') for record in records]
    if payload:
        return pl.DataFrame(payload, infer_schema_length=len(payload))
    return pl.DataFrame({field_name: [] for field_name in model_cls.model_fields})

def build_marts(mentions, threads, enrichment, ontology, *, dedup_clusters=None, author_quality=None):
    fact_mentions = _frame_from_models(mentions, MentionRecord)
    fact_entity_mentions = _frame_from_models(enrichment.entity_facts, EntityFact)
    fact_entity_candidate_clusters = _frame_from_models(enrichment.entity_candidate_clusters, EntityCandidateClusterFact)
    fact_threads = _frame_from_models(threads.summaries, ThreadSummary)
    fact_dedup_clusters = _frame_from_models(dedup_clusters or [], DedupClusterRecord)
    fact_topics = _frame_from_models(enrichment.topic_facts, TopicFact)
    fact_topic_artifacts = _frame_from_models(enrichment.topic_artifacts, TopicArtifactFact)
    fact_aspects = _frame_from_models(enrichment.aspect_opinion_facts, AspectOpinionFact)
    fact_sentiment = _frame_from_models(enrichment.sentiment_facts, SentimentFact)
    fact_target_sentiment = _frame_from_models(enrichment.target_sentiment_facts, TargetSentimentFact)
    fact_intents = _frame_from_models(enrichment.intent_facts, IntentFact)
    fact_issue_signals = _frame_from_models(enrichment.issue_signal_facts, IssueSignalFact)
    fact_author_quality = _frame_from_models(author_quality or [], AuthorQualityRecord)
    dim_entities = pl.DataFrame([{'canonical_entity_id': entity.id, 'entity_name': entity.name, 'entity_type': entity.entity_type, 'entity_kind': entity.entity_kind, 'knowledge_layer': entity.knowledge_layer, 'active_linking': entity.active_linking, 'taxonomy_ids': entity.taxonomy_ids} for entity in ontology.entities])
    dim_taxonomy = pl.DataFrame([node.model_dump(mode='json') for node in ontology.taxonomy_nodes])
    dim_sources = pl.DataFrame([item.model_dump(mode='json') for item in ontology.source_channels])
    dim_projects = fact_mentions.select(['project_id', 'task_id', 'platform']).unique().sort(['project_id', 'task_id']) if not fact_mentions.is_empty() else pl.DataFrame([])
    dim_time = fact_mentions.with_columns(pl.col('posted_at').str.slice(0, 10).str.to_date('%Y-%m-%d', strict=False).alias('date')).with_columns(pl.col('date').dt.year().alias('year'), pl.col('date').dt.month().alias('month'), pl.col('date').dt.week().alias('week'), pl.col('date').dt.weekday().alias('weekday')).select(['date', 'year', 'month', 'week', 'weekday']).unique().sort('date') if not fact_mentions.is_empty() else pl.DataFrame([])
    return MartBundle(tables={'fact_mentions': fact_mentions, 'fact_entity_mentions': fact_entity_mentions, 'fact_entity_candidate_clusters': fact_entity_candidate_clusters, 'fact_threads': fact_threads, 'fact_dedup_clusters': fact_dedup_clusters, 'fact_topics': fact_topics, 'fact_topic_artifacts': fact_topic_artifacts, 'fact_aspects': fact_aspects, 'fact_sentiment': fact_sentiment, 'fact_target_sentiment': fact_target_sentiment, 'fact_intents': fact_intents, 'fact_issue_signals': fact_issue_signals, 'fact_author_quality': fact_author_quality, 'dim_entities': dim_entities, 'dim_taxonomy': dim_taxonomy, 'dim_sources': dim_sources, 'dim_projects': dim_projects, 'dim_time': dim_time})
