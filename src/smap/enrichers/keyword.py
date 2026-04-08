from __future__ import annotations

from collections import Counter

from smap.enrichers.models import FactProvenance, KeywordFact
from smap.normalization.models import MentionRecord
from smap.threads.models import MentionContext

STOPWORDS = {
    "và",
    "là",
    "của",
    "cho",
    "the",
    "and",
    "with",
    "that",
    "this",
    "mấy",
    "bác",
}


class KeywordEnricher:
    name = "keyword"

    def enrich(self, mention: MentionRecord, context: MentionContext | None) -> list[KeywordFact]:
        tokens = [
            token
            for token in mention.normalized_text_compact.split()
            if len(token) > 2 and token not in STOPWORDS and token != "url"
        ]
        token_counts = Counter(tokens)
        keyphrases = [token for token, _ in token_counts.most_common(3)]
        keyphrases.extend(mention.hashtags[:3])
        keyphrases.extend(mention.keywords[:2])
        deduped: list[str] = []
        for keyphrase in keyphrases:
            value = keyphrase.strip().lower()
            if value and value not in deduped:
                deduped.append(value)
        return [
            KeywordFact(
                mention_id=mention.mention_id,
                source_uap_id=mention.source_uap_id,
                keyphrase=keyphrase,
                confidence=0.55 if keyphrase in mention.keywords else 0.42,
                provenance=FactProvenance(
                    source_uap_id=mention.source_uap_id,
                    mention_id=mention.mention_id,
                    provider_version="keyword-rules-v1",
                    rule_version="keyword-rules-v1",
                    evidence_text=context.context_text if context else mention.raw_text,
                ),
            )
            for keyphrase in deduped[:5]
        ]

