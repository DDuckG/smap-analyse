from __future__ import annotations

from smap.contracts.uap import (
    ParsedUAPRecord,
    content_keywords,
    content_links,
    content_title,
    engagement_saves,
)
from smap.normalization.models import MentionRecord, NormalizationBatch
from smap.normalization.text import normalize_social_text
from smap.providers.base import LanguageIdProvider


def normalize_record(
    record: ParsedUAPRecord,
    *,
    language_id_provider: LanguageIdProvider | None = None,
) -> MentionRecord:
    hashtags = getattr(record.content, "hashtags", None)
    language = getattr(record.content, "language", None)
    text_result = normalize_social_text(
        record.text,
        hashtags,
        language,
        language_id_provider=language_id_provider,
    )
    canonical_links = list(dict.fromkeys([*text_result.urls, *content_links(record)]))

    return MentionRecord(
        mention_id=record.identity.uap_id,
        source_uap_id=record.identity.uap_id,
        origin_id=record.identity.origin_id,
        uap_type=record.identity.uap_type,
        platform=record.identity.platform,
        project_id=record.identity.project_id,
        task_id=record.identity.task_id,
        root_id=record.hierarchy.root_id,
        parent_id=record.hierarchy.parent_id,
        depth=record.hierarchy.depth,
        author_id=record.author.id,
        author_username=record.author.username,
        author_nickname=record.author.nickname,
        raw_text=text_result.raw_text,
        normalized_text=text_result.normalized_text,
        normalized_text_compact=text_result.normalized_text_compact,
        language=text_result.language,
        language_confidence=text_result.language_confidence,
        language_provider=text_result.language_provider,
        language_provider_version=text_result.language_provider_version,
        language_model_id=text_result.language_model_id,
        language_source=text_result.language_source,
        language_metadata=text_result.language_metadata,
        language_supported=text_result.language_supported,
        language_rejection_reason=text_result.language_rejection_reason,
        text_quality_label=text_result.text_quality_label,
        text_quality_flags=text_result.text_quality_flags,
        text_quality_score=text_result.text_quality_score,
        mixed_language_uncertain=text_result.mixed_language_uncertain,
        semantic_route_hint=text_result.semantic_route_hint,
        hashtags=text_result.hashtags,
        keywords=content_keywords(record),
        urls=canonical_links,
        emojis=text_result.emojis,
        title=content_title(record),
        summary_title=content_title(record),
        source_url=record.identity.url,
        posted_at=record.temporal.posted_at,
        updated_at=record.temporal.updated_at,
        ingested_at=record.temporal.ingested_at,
        likes=getattr(record.engagement, "likes", None),
        comments_count=getattr(record.engagement, "comments_count", None),
        reply_count=getattr(record.engagement, "reply_count", None),
        shares=getattr(record.engagement, "shares", None),
        views=getattr(record.engagement, "views", None),
        saves=engagement_saves(record),
        bookmarks=engagement_saves(record),
        sort_score=getattr(record.engagement, "sort_score", None),
        is_shop_video=getattr(record.content, "is_shop_video", None),
    )


def normalize_batch(
    records: list[ParsedUAPRecord],
    *,
    language_id_provider: LanguageIdProvider | None = None,
) -> NormalizationBatch:
    normalized_mentions = [
        normalize_record(record, language_id_provider=language_id_provider)
        for record in records
    ]
    kept_mentions = [mention for mention in normalized_mentions if mention.language_supported]
    return NormalizationBatch(
        mentions=kept_mentions,
        filtered_out_records=len(normalized_mentions) - len(kept_mentions),
        filtered_out_unsupported_language=sum(1 for mention in normalized_mentions if not mention.language_supported),
    )
