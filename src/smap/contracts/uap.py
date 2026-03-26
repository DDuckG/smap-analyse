from __future__ import annotations
from datetime import datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class UAPType(StrEnum):
    POST = 'POST'
    COMMENT = 'COMMENT'
    REPLY = 'REPLY'

class Platform(StrEnum):
    TIKTOK = 'tiktok'
    FACEBOOK = 'facebook'
    YOUTUBE = 'youtube'

class MediaType(StrEnum):
    VIDEO = 'video'
    IMAGE = 'image'
    CAROUSEL = 'carousel'

class UAPBaseModel(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

class Identity(UAPBaseModel):
    uap_id: str
    origin_id: str
    uap_type: UAPType
    platform: Platform
    task_id: str
    project_id: str
    url: str | None = None

    @field_validator('uap_id', 'origin_id', 'task_id', 'project_id')
    @classmethod
    def validate_required_text(cls, value):
        if not value.strip():
            raise ValueError('value must not be empty')
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

class PostAuthor(Author):
    is_verified: bool | None = None
    account_type: str | None = None
    follower_count: int | None = None
    location_city: str | None = None

class PostContent(UAPBaseModel):
    text: str
    hashtags: list[str] | None = None
    tiktok_keywords: list[str] | None = None
    is_shop_video: bool | None = None
    music_title: str | None = None
    music_url: str | None = None
    summary_title: str | None = None
    subtitle_url: str | None = None
    language: str | None = None
    external_links: list[str] | None = None
    is_edited: bool | None = None
    detected_entities: list[str] | None = None

class CommentContent(UAPBaseModel):
    text: str
    external_links: list[str] | None = None

class PostEngagement(UAPBaseModel):
    likes: int | None = Field(default=None, ge=0)
    comments_count: int | None = Field(default=None, ge=0)
    shares: int | None = Field(default=None, ge=0)
    views: int | None = Field(default=None, ge=0)
    bookmarks: int | None = Field(default=None, ge=0)
    reply_count: int | None = Field(default=None, ge=0)
    discussion_depth: float | None = None
    velocity_score: float | None = None

class CommentEngagement(UAPBaseModel):
    likes: int | None = Field(default=None, ge=0)
    reply_count: int | None = Field(default=None, ge=0)
    sort_score: float | None = None

class ReplyEngagement(UAPBaseModel):
    likes: int | None = Field(default=None, ge=0)
    sort_score: float | None = None

class MediaItem(UAPBaseModel):
    type: MediaType
    url: str | None = None
    download_url: str | None = None
    duration: int | None = Field(default=None, ge=0)
    thumbnail: str | None = None

class Temporal(UAPBaseModel):
    posted_at: datetime | None = None
    ingested_at: datetime | None = None

class BaseUAPRecord(UAPBaseModel):
    identity: Identity
    hierarchy: Hierarchy
    content: PostContent | CommentContent
    author: Author | PostAuthor
    engagement: PostEngagement | CommentEngagement | ReplyEngagement
    temporal: Temporal

    @property
    def text(self):
        return self.content.text

    @property
    def root_id(self):
        return self.hierarchy.root_id

    @property
    def parent_id(self):
        return self.hierarchy.parent_id

class PostRecord(BaseUAPRecord):
    content: PostContent
    author: PostAuthor
    engagement: PostEngagement
    media: list[MediaItem] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_post(self):
        if self.identity.uap_type is not UAPType.POST:
            raise ValueError('post record must have identity.uap_type=POST')
        if self.hierarchy.parent_id is not None:
            raise ValueError('post parent_id must be null')
        if self.hierarchy.depth != 0:
            raise ValueError('post depth must be 0')
        if self.hierarchy.root_id != self.identity.uap_id:
            raise ValueError('post root_id must equal uap_id')
        if not self.identity.url:
            raise ValueError('post url is required')
        return self

class CommentRecord(BaseUAPRecord):
    content: CommentContent
    author: Author
    engagement: CommentEngagement

    @model_validator(mode='after')
    def validate_comment(self):
        if self.identity.uap_type is not UAPType.COMMENT:
            raise ValueError('comment record must have identity.uap_type=COMMENT')
        if self.hierarchy.depth != 1:
            raise ValueError('comment depth must be 1')
        if not self.hierarchy.parent_id:
            raise ValueError('comment parent_id is required')
        return self

class ReplyRecord(BaseUAPRecord):
    content: CommentContent
    author: Author
    engagement: ReplyEngagement

    @model_validator(mode='after')
    def validate_reply(self):
        if self.identity.uap_type is not UAPType.REPLY:
            raise ValueError('reply record must have identity.uap_type=REPLY')
        if self.hierarchy.depth < 2:
            raise ValueError('reply depth must be >= 2')
        if not self.hierarchy.parent_id:
            raise ValueError('reply parent_id is required')
        return self
ParsedUAPRecord = PostRecord | CommentRecord | ReplyRecord

def parse_uap_record(payload):
    identity = payload.get('identity')
    if not isinstance(identity, dict):
        raise ValueError('record.identity must be an object')
    uap_type = identity.get('uap_type')
    if uap_type == UAPType.POST:
        return PostRecord.model_validate(payload)
    if uap_type == UAPType.COMMENT:
        return CommentRecord.model_validate(payload)
    if uap_type == UAPType.REPLY:
        return ReplyRecord.model_validate(payload)
    raise ValueError(f'Unsupported UAP type: {uap_type!r}')
