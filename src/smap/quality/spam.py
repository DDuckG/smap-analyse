from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from smap.normalization.models import MentionRecord
from smap.quality.models import AuthorQualityRecord, QualityAnalysisResult

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class _MentionFeatures:
    mention_id: str
    mention_spam_score: float
    reason_codes: list[str]
    template_key: str
    repost_like: bool


class SpamScoringService:
    def __init__(
        self,
        *,
        mention_threshold: float = 0.65,
        author_threshold: float = 0.65,
        burst_window_minutes: int = 60,
        quality_weight_floor: float = 0.15,
        mention_discount_strength: float = 0.65,
        author_discount_strength: float = 0.45,
    ) -> None:
        self.mention_threshold = mention_threshold
        self.author_threshold = author_threshold
        self.burst_window_minutes = burst_window_minutes
        self.quality_weight_floor = quality_weight_floor
        self.mention_discount_strength = mention_discount_strength
        self.author_discount_strength = author_discount_strength

    def annotate(self, mentions: list[MentionRecord]) -> tuple[list[MentionRecord], QualityAnalysisResult]:
        if not mentions:
            return [], QualityAnalysisResult()
        mention_features = {mention.mention_id: self._mention_features(mention) for mention in mentions}
        template_counts = Counter(feature.template_key for feature in mention_features.values() if feature.template_key)
        author_groups: dict[str, list[MentionRecord]] = defaultdict(list)
        for mention in mentions:
            author_groups[mention.author_id].append(mention)

        author_profiles: dict[str, AuthorQualityRecord] = {}
        for author_id, author_mentions in author_groups.items():
            author_profiles[author_id] = self._author_profile(
                author_id,
                author_mentions,
                mention_features=mention_features,
                template_counts=template_counts,
            )

        updated_mentions: list[MentionRecord] = []
        for mention in mentions:
            feature = mention_features[mention.mention_id]
            profile = author_profiles[mention.author_id]
            suspicious = feature.mention_spam_score >= self.mention_threshold
            author_suspicious = profile.author_suspicious
            quality_weight = self._quality_weight(
                dedup_weight=mention.dedup_weight,
                mention_spam_score=feature.mention_spam_score,
                author_inorganic_score=profile.author_inorganic_score,
            )
            text_quality_flags = list(
                dict.fromkeys(
                    [
                        *mention.text_quality_flags,
                        *(["duplicate_like"] if mention.dedup_cluster_size > 1 else []),
                        *(["spam_like"] if suspicious or author_suspicious else []),
                    ]
                )
            )
            semantic_route_hint = (
                "semantic_lite"
                if any(
                    flag in {"reaction_only", "low_information", "duplicate_like", "spam_like", "url_heavy"}
                    for flag in text_quality_flags
                )
                else "semantic_full"
            )
            updated_mentions.append(
                mention.model_copy(
                    update={
                        "mention_spam_score": feature.mention_spam_score,
                        "author_inorganic_score": profile.author_inorganic_score,
                        "mention_suspicious": suspicious,
                        "author_suspicious": author_suspicious,
                        "suspicion_reason_codes": [
                            *feature.reason_codes,
                            *(profile.reason_codes if author_suspicious else []),
                        ],
                        "quality_weight": quality_weight,
                        "text_quality_flags": text_quality_flags,
                        "text_quality_label": (
                            "spam_like"
                            if "spam_like" in text_quality_flags
                            else "duplicate_like"
                            if "duplicate_like" in text_quality_flags and mention.text_quality_label == "normal"
                            else mention.text_quality_label
                        ),
                        "text_quality_score": round(min(mention.text_quality_score, quality_weight), 4),
                        "semantic_route_hint": semantic_route_hint,
                    }
                )
            )
        return (
            updated_mentions,
            QualityAnalysisResult(
                mentions_updated=len(updated_mentions),
                author_profiles=sorted(
                    author_profiles.values(),
                    key=lambda item: (-item.author_inorganic_score, -item.mention_count, item.author_id),
                ),
            ),
        )

    def _mention_features(self, mention: MentionRecord) -> _MentionFeatures:
        tokens = list(_TOKEN_RE.findall(mention.normalized_text_compact))
        token_count = len(tokens)
        lexical_diversity = (len(set(tokens)) / token_count) if token_count else 1.0
        repetition_ratio = (1.0 - lexical_diversity) if token_count else 0.0
        max_token_share = max(Counter(tokens).values(), default=0) / token_count if token_count else 0.0
        url_density = len(mention.urls) / max(token_count, 1)
        duplicate_pressure = max(mention.dedup_cluster_size - 1, 0) / max(mention.dedup_cluster_size, 1)
        template_key = self._template_key(mention.normalized_text)
        repost_like = self._is_repost_like(mention)
        score_components = {
            "high_url_density": min(url_density * 2.5, 1.0),
            "low_lexical_diversity": max(0.0, 0.45 - lexical_diversity) / 0.45 if token_count >= 4 else 0.0,
            "high_repetition_ratio": max(repetition_ratio, max_token_share),
            "duplicate_cluster_participation": duplicate_pressure,
            "repost_like_behavior": 1.0 if repost_like else 0.0,
        }
        reason_codes = [name for name, score in score_components.items() if score >= 0.5]
        score = round(
            min(
                (
                    score_components["high_url_density"] * 0.22
                    + score_components["low_lexical_diversity"] * 0.22
                    + score_components["high_repetition_ratio"] * 0.20
                    + score_components["duplicate_cluster_participation"] * 0.22
                    + score_components["repost_like_behavior"] * 0.14
                ),
                1.0,
            ),
            4,
        )
        return _MentionFeatures(
            mention_id=mention.mention_id,
            mention_spam_score=score,
            reason_codes=reason_codes,
            template_key=template_key,
            repost_like=repost_like,
        )

    def _author_profile(
        self,
        author_id: str,
        mentions: list[MentionRecord],
        *,
        mention_features: dict[str, _MentionFeatures],
        template_counts: Counter[str],
    ) -> AuthorQualityRecord:
        mention_scores = [mention_features[mention.mention_id].mention_spam_score for mention in mentions]
        suspicious_mentions = [score for score in mention_scores if score >= self.mention_threshold]
        duplicate_mentions = [mention for mention in mentions if mention.dedup_cluster_size > 1]
        repost_count = sum(1 for mention in mentions if mention_features[mention.mention_id].repost_like)
        template_share = self._template_ratio(mentions, mention_features, template_counts)
        burstiness = self._burstiness_score(mentions)
        duplicate_rate = len(duplicate_mentions) / max(len(mentions), 1)
        average_score = sum(mention_scores) / max(len(mention_scores), 1)
        score = round(
            min(
                average_score * 0.42
                + burstiness * 0.18
                + duplicate_rate * 0.20
                + template_share * 0.12
                + (repost_count / max(len(mentions), 1)) * 0.08,
                1.0,
            ),
            4,
        )
        reason_codes: list[str] = []
        if burstiness >= 0.55:
            reason_codes.append("author_burstiness")
        if duplicate_rate >= 0.45:
            reason_codes.append("author_duplicate_participation")
        if template_share >= 0.45:
            reason_codes.append("author_template_reuse")
        if repost_count / max(len(mentions), 1) >= 0.4:
            reason_codes.append("author_repost_like_ratio")
        if average_score >= 0.55:
            reason_codes.append("high_average_mention_spam_score")
        score = round(min(score + min(len(reason_codes) * 0.03, 0.09), 1.0), 4)
        return AuthorQualityRecord(
            author_id=author_id,
            mention_count=len(mentions),
            suspicious_mention_count=len(suspicious_mentions),
            author_inorganic_score=score,
            author_suspicious=score >= self.author_threshold,
            burstiness_score=burstiness,
            template_ratio=round(template_share, 4),
            duplicate_cluster_participation_rate=round(duplicate_rate, 4),
            repost_like_ratio=round(repost_count / max(len(mentions), 1), 4),
            average_mention_spam_score=round(average_score, 4),
            reason_codes=reason_codes,
        )

    def _template_key(self, normalized_text: str) -> str:
        compact = re.sub(r"<url>", " <url> ", normalized_text)
        compact = re.sub(r"#[\w_]+", " <tag> ", compact)
        compact = re.sub(r"\d+", " <num> ", compact)
        compact = re.sub(r"\s+", " ", compact).strip()
        return compact

    def _template_ratio(
        self,
        mentions: list[MentionRecord],
        mention_features: dict[str, _MentionFeatures],
        template_counts: Counter[str],
    ) -> float:
        if not mentions:
            return 0.0
        repeat_hits = sum(
            1
            for mention in mentions
            if template_counts.get(mention_features[mention.mention_id].template_key, 0) >= 3
        )
        return repeat_hits / len(mentions)

    def _burstiness_score(self, mentions: list[MentionRecord]) -> float:
        posted = sorted(
            mention.posted_at
            for mention in mentions
            if isinstance(mention.posted_at, datetime)
        )
        if len(posted) < 2:
            return 0.0
        window = timedelta(minutes=self.burst_window_minutes)
        best = 1
        left = 0
        for right, current in enumerate(posted):
            while current - posted[left] > window:
                left += 1
            best = max(best, right - left + 1)
        return round(min(best / max(len(posted), 1), 1.0), 4)

    def _is_repost_like(self, mention: MentionRecord) -> bool:
        text = mention.normalized_text_compact
        return text.startswith("rt ") or text.startswith("repost ") or text.startswith("shared from ")

    def _quality_weight(
        self,
        *,
        dedup_weight: float,
        mention_spam_score: float,
        author_inorganic_score: float,
    ) -> float:
        mention_multiplier = max(1.0 - (mention_spam_score * self.mention_discount_strength), self.quality_weight_floor)
        author_multiplier = max(1.0 - (author_inorganic_score * self.author_discount_strength), self.quality_weight_floor)
        return round(dedup_weight * mention_multiplier * author_multiplier, 6)
