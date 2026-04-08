from __future__ import annotations

from smap.enrichers.models import FactProvenance, IntentFact
from smap.normalization.models import MentionRecord
from smap.threads.models import MentionContext


class IntentEnricher:
    name = "intent"

    def enrich(self, mention: MentionRecord, context: MentionContext | None) -> list[IntentFact]:
        text = mention.normalized_text
        if "?" in mention.raw_text or "hỏi" in text:
            intent = "question"
            confidence = 0.85
        elif any(token in text for token in ("mua", "đặt", "giá", "buy")):
            intent = "purchase_intent"
            confidence = 0.7
        elif any(token in text for token in ("đẹp", "tốt", "love", "chúc mừng")):
            intent = "praise"
            confidence = 0.7
        elif any(token in text for token in ("lỗi", "chậm", "bad", "không")):
            intent = "complaint"
            confidence = 0.65
        elif any(token in text for token in ("so với", "compare", "hơn")):
            intent = "compare"
            confidence = 0.6
        else:
            intent = "commentary"
            confidence = 0.4
        return [
            IntentFact(
                mention_id=mention.mention_id,
                source_uap_id=mention.source_uap_id,
                intent=intent,
                confidence=confidence,
                provenance=FactProvenance(
                    source_uap_id=mention.source_uap_id,
                    mention_id=mention.mention_id,
                    provider_version="intent-rules-v1",
                    rule_version="intent-rules-v1",
                    evidence_text=context.context_text if context else mention.raw_text,
                ),
            )
        ]
