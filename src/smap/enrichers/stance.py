from __future__ import annotations
from smap.enrichers.models import FactProvenance, StanceFact, StanceLabel
from smap.normalization.models import MentionRecord
from smap.threads.models import MentionContext

class StanceEnricher:
    name = 'stance'

    def enrich(self, mention, context):
        text = mention.normalized_text
        stance: StanceLabel
        if '?' in mention.raw_text or any((token in text for token in ('hỏi', 'sao', 'why', 'what'))):
            stance = 'question'
            confidence = 0.8
        elif any((token in text for token in ('không', 'lỗi', 'bad', 'hate', 'chậm'))):
            stance = 'oppose'
            confidence = 0.65
        elif any((token in text for token in ('đẹp', 'tốt', 'love', 'good', 'tự hào'))):
            stance = 'support'
            confidence = 0.65
        else:
            stance = 'neutral'
            confidence = 0.45
        return [StanceFact(mention_id=mention.mention_id, source_uap_id=mention.source_uap_id, stance=stance, confidence=confidence, provenance=FactProvenance(source_uap_id=mention.source_uap_id, mention_id=mention.mention_id, provider_version='stance-rules-v1', rule_version='stance-rules-v1', evidence_text=context.context_text if context else mention.raw_text))]
