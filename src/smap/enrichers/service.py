from __future__ import annotations
from smap.enrichers.entity import EntityExtractionEnricher
from smap.enrichers.intent import IntentEnricher
from smap.enrichers.keyword import KeywordEnricher
from smap.enrichers.models import EnrichmentBundle
from smap.enrichers.semantic import SemanticInferenceEnricher
from smap.enrichers.source import SourceInfluenceEnricher
from smap.enrichers.stance import StanceEnricher
from smap.enrichers.topic import TopicCandidateEnricher
from smap.normalization.models import MentionRecord
from smap.threads.models import MentionContext

class EnricherService:

    def __init__(self, entity_enricher, *, topic_enricher=None, semantic_enricher=None):
        self.entity_enricher = entity_enricher
        self.keyword_enricher = KeywordEnricher()
        self.topic_enricher = topic_enricher or TopicCandidateEnricher()
        self.semantic_enricher = semantic_enricher or SemanticInferenceEnricher()
        self.stance_enricher = StanceEnricher()
        self.intent_enricher = IntentEnricher()
        self.source_enricher = SourceInfluenceEnricher()

    def enrich_mentions(self, mentions, contexts):
        context_map = {context.mention_id: context for context in contexts}
        bundle = EnrichmentBundle()
        self.entity_enricher.prepare(mentions, contexts)
        for mention in mentions:
            context = context_map.get(mention.mention_id)
            bundle.entity_facts.extend(self.entity_enricher.enrich(mention, context))
            bundle.stance_facts.extend(self.stance_enricher.enrich(mention, context))
            bundle.intent_facts.extend(self.intent_enricher.enrich(mention, context))
            bundle.source_influence_facts.extend(self.source_enricher.enrich(mention, context))
        bundle.entity_facts, bundle.entity_candidate_clusters = self.entity_enricher.annotate_batch_local_candidates(bundle.entity_facts, mentions)
        for mention in mentions:
            context = context_map.get(mention.mention_id)
            bundle.keyword_facts.extend(self.keyword_enricher.enrich(mention, context))
        semantic_result = self.semantic_enricher.enrich(mentions, contexts, bundle.entity_facts)
        bundle.sentiment_facts.extend(semantic_result.mention_sentiments)
        bundle.target_sentiment_facts.extend(semantic_result.target_sentiments)
        bundle.aspect_opinion_facts.extend(semantic_result.aspect_opinions)
        bundle.issue_signal_facts.extend(semantic_result.issue_signals)
        self.topic_enricher.prepare(mentions, contexts, entity_facts=bundle.entity_facts, aspect_facts=bundle.aspect_opinion_facts, issue_facts=bundle.issue_signal_facts)
        for mention in mentions:
            context = context_map.get(mention.mention_id)
            bundle.topic_facts.extend(self.topic_enricher.enrich(mention, context))
        bundle.topic_artifacts.extend(self.topic_enricher.artifacts())
        return bundle
