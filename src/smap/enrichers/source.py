from __future__ import annotations
from smap.enrichers.models import FactProvenance, InfluenceTier, SourceInfluenceFact
from smap.normalization.models import MentionRecord
from smap.threads.models import MentionContext

class SourceInfluenceEnricher:
    name = 'source_influence'

    def enrich(self, mention, context):
        engagement_score = float((mention.likes or 0) + (mention.comments_count or 0) + (mention.reply_count or 0) + (mention.shares or 0) + (mention.views or 0) / 100.0)
        tier: InfluenceTier
        if engagement_score >= 500:
            tier = 'macro'
        elif engagement_score >= 100:
            tier = 'mid'
        elif engagement_score >= 20:
            tier = 'micro'
        else:
            tier = 'nano'
        return [SourceInfluenceFact(mention_id=mention.mention_id, source_uap_id=mention.source_uap_id, author_id=mention.author_id, channel=mention.platform.value, influence_tier=tier, engagement_score=round(engagement_score, 3), confidence=0.7, provenance=FactProvenance(source_uap_id=mention.source_uap_id, mention_id=mention.mention_id, provider_version='source-influence-v1', rule_version='source-influence-v1', evidence_text=context.context_text if context else mention.raw_text))]
