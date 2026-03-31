from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class UAPType(StrEnum):
    POST = "POST"
    COMMENT = "COMMENT"
    REPLY = "REPLY"


class Platform(StrEnum):
    TIKTOK = "tiktok"
    FACEBOOK = "facebook"
    YOUTUBE = "youtube"


class MediaType(StrEnum):
    VIDEO = "video"
    IMAGE = "image"
    CAROUSEL = "carousel"


class UAPBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class Identity(UAPBaseModel):
    uap_id: str
    origin_id: str
    uap_type: UAPType
    platform: Platform
    task_id: str
    project_id: str
    url: str | None = None

    @field_validator("uap_id", "origin_id", "task_id", "project_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class Hierarchy(UAPBaseModel):
    parent_id: str | None = None
    root_id: str
    depth: int = Field(ge=0)


class Author(UAPBaseModel):
    id: str
    username: str | None = None
    nickname: str | None = None
    avatar: str | None = None
    profile_url: str | None = None


class PostAuthor(Author):
    is_verified: bool | None = None
    account_type: str | None = None
    follower_count: int | None = None
    location_city: str | None = None


class ContentBase(UAPBaseModel):
    text: str
    title: str | None = None
    subtitle: str | None = None
    hashtags: list[str] | None = None
    keywords: list[str] | None = None
    tiktok_keywords: list[str] | None = None
    links: list[str] | None = None
    external_links: list[str] | None = None
    language: str | None = None

    @property
    def canonical_title(self) -> str | None:
        return self.title

    @property
    def canonical_keywords(self) -> list[str]:
        return list(self.keywords or self.tiktok_keywords or [])

    @property
    def canonical_links(self) -> list[str]:
        return list(self.links or self.external_links or [])


class PostContent(ContentBase):
    is_shop_video: bool | None = None
    music_title: str | None = None
    music_url: str | None = None
    summary_title: str | None = None
    subtitle_url: str | None = None
    is_edited: bool | None = None
    detected_entities: list[str] | None = None

    @property
    def canonical_title(self) -> str | None:
        return self.title or self.summary_title


class CommentContent(ContentBase):
    @property
    def canonical_title(self) -> str | None:
        return self.title


class PostEngagement(UAPBaseModel):
    likes: int | None = Field(default=None, ge=0)
    comments_count: int | None = Field(default=None, ge=0)
    shares: int | None = Field(default=None, ge=0)
    views: int | None = Field(default=None, ge=0)
    saves: int | None = Field(default=None, ge=0)
    bookmarks: int | None = Field(default=None, ge=0)
    reply_count: int | None = Field(default=None, ge=0)
    discussion_depth: float | None = None
    velocity_score: float | None = None

    @property
    def canonical_saves(self) -> int | None:
        return self.saves if self.saves is not None else self.bookmarks


class CommentEngagement(UAPBaseModel):
    likes: int | None = Field(default=None, ge=0)
    saves: int | None = Field(default=None, ge=0)
    reply_count: int | None = Field(default=None, ge=0)
    sort_score: float | None = None


class ReplyEngagement(UAPBaseModel):
    likes: int | None = Field(default=None, ge=0)
    saves: int | None = Field(default=None, ge=0)
    sort_score: float | None = None


class MediaItem(UAPBaseModel):
    type: MediaType
    url: str | None = None
    download_url: str | None = None
    duration: int | None = Field(default=None, ge=0)
    thumbnail: str | None = None
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)


class Temporal(UAPBaseModel):
    posted_at: datetime | None = None
    updated_at: datetime | None = None
    ingested_at: datetime | None = None


class BaseUAPRecord(UAPBaseModel):
    identity: Identity
    hierarchy: Hierarchy
    content: PostContent | CommentContent
    author: Author | PostAuthor
    engagement: PostEngagement | CommentEngagement | ReplyEngagement
    temporal: Temporal
    platform_meta: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return self.content.text

    @property
    def root_id(self) -> str:
        return self.hierarchy.root_id

    @property
    def parent_id(self) -> str | None:
        return self.hierarchy.parent_id


class PostRecord(BaseUAPRecord):
    content: PostContent
    author: PostAuthor
    engagement: PostEngagement
    media: list[MediaItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_post(self) -> PostRecord:
        if self.identity.uap_type is not UAPType.POST:
            raise ValueError("post record must have identity.uap_type=POST")
        if self.hierarchy.parent_id is not None:
            raise ValueError("post parent_id must be null")
        if self.hierarchy.depth != 0:
            raise ValueError("post depth must be 0")
        if self.hierarchy.root_id != self.identity.uap_id:
            raise ValueError("post root_id must equal uap_id")
        if not self.identity.url:
            raise ValueError("post url is required")
        return self


class CommentRecord(BaseUAPRecord):
    content: CommentContent
    author: Author
    engagement: CommentEngagement

    @model_validator(mode="after")
    def validate_comment(self) -> CommentRecord:
        if self.identity.uap_type is not UAPType.COMMENT:
            raise ValueError("comment record must have identity.uap_type=COMMENT")
        if self.hierarchy.depth != 1:
            raise ValueError("comment depth must be 1")
        if not self.hierarchy.parent_id:
            raise ValueError("comment parent_id is required")
        return self


class ReplyRecord(BaseUAPRecord):
    content: CommentContent
    author: Author
    engagement: ReplyEngagement

    @model_validator(mode="after")
    def validate_reply(self) -> ReplyRecord:
        if self.identity.uap_type is not UAPType.REPLY:
            raise ValueError("reply record must have identity.uap_type=REPLY")
        if self.hierarchy.depth < 2:
            raise ValueError("reply depth must be >= 2")
        if not self.hierarchy.parent_id:
            raise ValueError("reply parent_id is required")
        return self


ParsedUAPRecord = PostRecord | CommentRecord | ReplyRecord

_OPTIONAL_EMPTY_STRING_PATHS: tuple[tuple[str, ...], ...] = (
    ("identity", "url"),
    ("hierarchy", "parent_id"),
    ("author", "username"),
    ("author", "nickname"),
    ("author", "avatar"),
    ("author", "profile_url"),
    ("author", "account_type"),
    ("author", "location_city"),
    ("content", "title"),
    ("content", "subtitle"),
    ("content", "music_title"),
    ("content", "music_url"),
    ("content", "summary_title"),
    ("content", "subtitle_url"),
    ("content", "language"),
    ("temporal", "posted_at"),
    ("temporal", "updated_at"),
    ("temporal", "ingested_at"),
)


def _normalize_optional_empty_strings(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for path in _OPTIONAL_EMPTY_STRING_PATHS:
        source: Any = payload
        for key in path[:-1]:
            if not isinstance(source, dict):
                source = None
                break
            source = source.get(key)
        if not isinstance(source, dict):
            continue
        value = source.get(path[-1])
        if not isinstance(value, str) or value.strip():
            continue
        target: dict[str, Any] | None = result
        for key in path[:-1]:
            if target is None:
                break
            child = target.get(key)
            if not isinstance(child, dict):
                target = None
                break
            child_copy = dict(child)
            target[key] = child_copy
            target = child_copy
        if isinstance(target, dict):
            target[path[-1]] = None
    return result


def _clone_child_mapping(
    result: dict[str, Any],
    payload: dict[str, Any],
    key: str,
) -> dict[str, Any] | None:
    source = payload.get(key)
    if not isinstance(source, dict):
        return None
    cloned = result.get(key)
    if not isinstance(cloned, dict):
        cloned = dict(source)
        result[key] = cloned
    return cloned


def _has_non_empty_items(value: Any) -> bool:
    return isinstance(value, list) and any(item not in (None, "") for item in value)


def _has_non_blank_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_uap_aliases(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    content = _clone_child_mapping(result, payload, "content")
    engagement = _clone_child_mapping(result, payload, "engagement")
    platform_meta = payload.get("platform_meta")
    tiktok_meta = (
        platform_meta.get("tiktok")
        if isinstance(platform_meta, dict) and isinstance(platform_meta.get("tiktok"), dict)
        else None
    )

    if content is not None:
        if not _has_non_blank_text(content.get("title")) and _has_non_blank_text(content.get("summary_title")):
            content["title"] = content["summary_title"]
        if not _has_non_blank_text(content.get("summary_title")) and _has_non_blank_text(content.get("title")):
            content["summary_title"] = content["title"]
        if not _has_non_empty_items(content.get("keywords")) and _has_non_empty_items(content.get("tiktok_keywords")):
            content["keywords"] = list(content["tiktok_keywords"])
        if not _has_non_empty_items(content.get("tiktok_keywords")) and _has_non_empty_items(content.get("keywords")):
            content["tiktok_keywords"] = list(content["keywords"])
        if not _has_non_empty_items(content.get("links")) and _has_non_empty_items(content.get("external_links")):
            content["links"] = list(content["external_links"])
        if not _has_non_empty_items(content.get("external_links")) and _has_non_empty_items(content.get("links")):
            content["external_links"] = list(content["links"])
        if isinstance(tiktok_meta, dict):
            if not _has_non_blank_text(content.get("music_title")) and _has_non_blank_text(tiktok_meta.get("music_title")):
                content["music_title"] = tiktok_meta["music_title"]
            if not _has_non_blank_text(content.get("music_url")) and _has_non_blank_text(tiktok_meta.get("music_url")):
                content["music_url"] = tiktok_meta["music_url"]
            if content.get("is_shop_video") is None and isinstance(tiktok_meta.get("is_shop_video"), bool):
                content["is_shop_video"] = tiktok_meta["is_shop_video"]

    if engagement is not None:
        if engagement.get("saves") is None and engagement.get("bookmarks") is not None:
            engagement["saves"] = engagement["bookmarks"]
        if engagement.get("bookmarks") is None and engagement.get("saves") is not None:
            engagement["bookmarks"] = engagement["saves"]
        if engagement.get("sort_score") is None and isinstance(tiktok_meta, dict) and tiktok_meta.get("sort_score") is not None:
            engagement["sort_score"] = tiktok_meta["sort_score"]
    return result


def parse_uap_record(payload: dict[str, Any]) -> ParsedUAPRecord:
    normalized_payload = _normalize_uap_aliases(_normalize_optional_empty_strings(payload))
    identity = normalized_payload.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("record.identity must be an object")
    uap_type = identity.get("uap_type")
    if uap_type == UAPType.POST:
        return PostRecord.model_validate(normalized_payload)
    if uap_type == UAPType.COMMENT:
        return CommentRecord.model_validate(normalized_payload)
    if uap_type == UAPType.REPLY:
        return ReplyRecord.model_validate(normalized_payload)
    raise ValueError(f"Unsupported UAP type: {uap_type!r}")


def content_title(record: ParsedUAPRecord) -> str | None:
    return record.content.canonical_title


def content_keywords(record: ParsedUAPRecord) -> list[str]:
    return record.content.canonical_keywords


def content_links(record: ParsedUAPRecord) -> list[str]:
    return record.content.canonical_links


def engagement_saves(record: ParsedUAPRecord) -> int | None:
    engagement = record.engagement
    if isinstance(engagement, PostEngagement):
        return engagement.canonical_saves
    return getattr(engagement, "saves", None)
