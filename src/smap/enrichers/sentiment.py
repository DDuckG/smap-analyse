from __future__ import annotations

from smap.enrichers.models import FactProvenance, SentimentFact, SentimentLabel
from smap.normalization.models import MentionRecord
from smap.threads.models import MentionContext

POSITIVE_WORDS = {
    "tốt",
    "đẹp",
    "ổn",
    "ngon",
    "xịn",
    "thích",
    "nice",
    "great",
    "love",
    "good",
    "tự hào",
}
NEGATIVE_WORDS = {
    "tệ",
    "xấu",
    "chán",
    "kém",
    "lỗi",
    "chậm",
    "ghét",
    "bad",
    "slow",
    "hate",
    "không",
    "khó",
}


class SentimentEnricher:
    name = "sentiment"

    def enrich(self, mention: MentionRecord, context: MentionContext | None) -> list[SentimentFact]:
        text = mention.normalized_text
        positive_hits = sum(1 for word in POSITIVE_WORDS if word in text)
        negative_hits = sum(1 for word in NEGATIVE_WORDS if word in text)

        sentiment: SentimentLabel
        if positive_hits and negative_hits:
            sentiment = "mixed"
        elif positive_hits > negative_hits:
            sentiment = "positive"
        elif negative_hits > positive_hits:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        magnitude = abs(positive_hits - negative_hits)
        score = magnitude / max(positive_hits + negative_hits, 1)
        confidence = 0.45 + min(0.45, (positive_hits + negative_hits) * 0.1)

        return [
            SentimentFact(
                mention_id=mention.mention_id,
                source_uap_id=mention.source_uap_id,
                sentiment=sentiment,
                score=round(score, 3),
                confidence=round(confidence, 3),
                provenance=FactProvenance(
                    source_uap_id=mention.source_uap_id,
                    mention_id=mention.mention_id,
                    provider_version="sentiment-lexicon-v1",
                    rule_version="sentiment-lexicon-v1",
                    evidence_text=mention.raw_text,
                ),
            )
        ]
