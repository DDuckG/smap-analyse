from __future__ import annotations

from smap.enrichers.models import AspectOpinionFact, FactProvenance
from smap.normalization.models import MentionRecord
from smap.threads.models import MentionContext

ASPECT_KEYWORDS = {
    "price": {"giá", "price", "rẻ", "đắt"},
    "quality": {"chất lượng", "quality", "ổn", "kém"},
    "design": {"đẹp", "thiết kế", "design"},
    "performance": {"mạnh", "performance", "vận hành", "ổn"},
    "battery": {"pin", "battery"},
    "charging": {"sạc", "charging"},
    "safety": {"an toàn", "safety"},
    "trust": {"tin", "trust", "uy tín"},
    "delivery": {"giao", "delivery"},
}


class AspectOpinionEnricher:
    name = "aspect_opinion"

    def enrich(self, mention: MentionRecord, context: MentionContext | None) -> list[AspectOpinionFact]:
        text = mention.normalized_text
        facts: list[AspectOpinionFact] = []
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                sentiment = (
                    "negative"
                    if any(marker in text for marker in ("không", "chậm", "kém", "bad"))
                    else "positive"
                    if any(marker in text for marker in ("đẹp", "tốt", "ổn", "good"))
                    else "neutral"
                )
                facts.append(
                    AspectOpinionFact(
                        mention_id=mention.mention_id,
                        source_uap_id=mention.source_uap_id,
                        aspect=aspect,
                        opinion_text=mention.raw_text[:140],
                        sentiment=sentiment,
                        confidence=0.62,
                        provenance=FactProvenance(
                            source_uap_id=mention.source_uap_id,
                            mention_id=mention.mention_id,
                            provider_version="aspect-rules-v1",
                            rule_version="aspect-rules-v1",
                            evidence_text=context.context_text if context else mention.raw_text,
                        ),
                    )
                )
        return facts
