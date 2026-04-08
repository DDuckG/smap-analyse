from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from smap.normalization.models import MentionRecord
from smap.ontology.models import OntologyRegistry
from smap.review.knowledge_state_hashing import (
    non_review_overlay_fingerprint,
    ontology_semantic_fingerprint,
    reviewed_overlay_fingerprint,
    stable_content_hash,
)
from smap.review.types import ReviewProblemClass, ReviewScopeLevel, SemanticMatchMode

REVIEW_SIGNATURE_VERSION = 4


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    return stable_content_hash(payload)


def _language_family(language: str | None) -> str | None:
    if language is None or not language.strip():
        return None
    return language.split("-")[0].strip().lower() or None


def build_ontology_fingerprint(registry: OntologyRegistry) -> str:
    return ontology_semantic_fingerprint(registry)


def build_overlay_fingerprint(registry: OntologyRegistry) -> str:
    return non_review_overlay_fingerprint(registry)


def build_reviewed_overlay_fingerprint(registry: OntologyRegistry) -> str:
    return reviewed_overlay_fingerprint(registry)


class StaticScopeKey(BaseModel):
    project_id: str | None = None
    task_id: str | None = None
    platform: str | None = None
    root_id: str | None = None
    language_family: str | None = None
    signature_version: int = REVIEW_SIGNATURE_VERSION

    def fingerprint(self) -> str:
        return _stable_fingerprint(self.model_dump(mode="json"))


class KnowledgeStateFingerprint(BaseModel):
    ontology_fingerprint: str | None = None
    overlay_fingerprint: str | None = None
    reviewed_overlay_fingerprint: str | None = None
    signature_version: int = REVIEW_SIGNATURE_VERSION

    def fingerprint(self) -> str:
        return _stable_fingerprint(self.model_dump(mode="json"))


class ReviewContext(BaseModel):
    project_id: str | None = None
    task_id: str | None = None
    platform: str | None = None
    root_id: str | None = None
    language: str | None = None
    language_family: str | None = None
    ontology_fingerprint: str = "unknown"
    overlay_fingerprint: str = "unknown"
    reviewed_overlay_fingerprint: str = "none"
    signature_version: int = REVIEW_SIGNATURE_VERSION
    source_family: str | None = None
    source_domain: str | None = None

    def fingerprint(self) -> str:
        return _stable_fingerprint(self.model_dump(mode="json"))

    def static_scope_key(self) -> StaticScopeKey:
        return StaticScopeKey(
            project_id=self.project_id,
            task_id=self.task_id,
            platform=self.platform,
            root_id=self.root_id,
            language_family=self.language_family,
            signature_version=self.signature_version,
        )

    def knowledge_state(self) -> KnowledgeStateFingerprint:
        return KnowledgeStateFingerprint(
            ontology_fingerprint=self.ontology_fingerprint,
            overlay_fingerprint=self.overlay_fingerprint,
            reviewed_overlay_fingerprint=self.reviewed_overlay_fingerprint,
            signature_version=self.signature_version,
        )

    @classmethod
    def from_mention(
        cls,
        mention: MentionRecord,
        registry: OntologyRegistry,
        *,
        source_family: str | None = None,
        source_domain: str | None = None,
    ) -> ReviewContext:
        return cls(
            project_id=mention.project_id,
            task_id=mention.task_id,
            platform=str(mention.platform),
            root_id=mention.root_id,
            language=mention.language,
            language_family=_language_family(mention.language),
            ontology_fingerprint=build_ontology_fingerprint(registry),
            overlay_fingerprint=build_overlay_fingerprint(registry),
            reviewed_overlay_fingerprint=build_reviewed_overlay_fingerprint(registry),
            signature_version=REVIEW_SIGNATURE_VERSION,
            source_family=source_family,
            source_domain=source_domain,
        )


class SemanticSignature(BaseModel):
    item_type: str
    problem_class: ReviewProblemClass
    normalized_candidate_text: str | None = None
    entity_type_hint: str | None = None
    ambiguity_signature: str | None = None
    candidate_canonical_ids: list[str] = Field(default_factory=list)
    label_key: str | None = None
    signature_version: int = REVIEW_SIGNATURE_VERSION

    def fingerprint(self) -> str:
        return _stable_fingerprint(self.model_dump(mode="json"))

    def match_value(self, match_mode: SemanticMatchMode) -> str:
        if match_mode == SemanticMatchMode.AMBIGUITY_SET:
            return "::".join([self.normalized_candidate_text or "-", self.ambiguity_signature or "-"])
        if match_mode == SemanticMatchMode.CLASSIFICATION_LABEL:
            return self.label_key or "-"
        if match_mode == SemanticMatchMode.NORMALIZED_FORM_AND_TYPE:
            return "::".join([self.normalized_candidate_text or "-", self.entity_type_hint or "-"])
        return self.normalized_candidate_text or self.label_key or "-"


class ScopeKey(BaseModel):
    scope_level: ReviewScopeLevel
    static_scope: StaticScopeKey = Field(default_factory=StaticScopeKey)
    knowledge_state: KnowledgeStateFingerprint = Field(default_factory=KnowledgeStateFingerprint)
    signature_version: int = REVIEW_SIGNATURE_VERSION

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_payload(cls, value: object) -> object:
        if not isinstance(value, dict) or "static_scope" in value or "knowledge_state" in value:
            return value
        legacy = dict(value)
        return {
            "scope_level": legacy.get("scope_level"),
            "static_scope": {
                "project_id": legacy.get("project_id"),
                "task_id": legacy.get("task_id"),
                "platform": legacy.get("platform"),
                "root_id": legacy.get("root_id"),
                "language_family": legacy.get("language_family"),
                "signature_version": legacy.get("signature_version", REVIEW_SIGNATURE_VERSION),
            },
            "knowledge_state": {
                "ontology_fingerprint": legacy.get("ontology_fingerprint"),
                "overlay_fingerprint": legacy.get("overlay_fingerprint"),
                "reviewed_overlay_fingerprint": legacy.get("reviewed_overlay_fingerprint"),
                "signature_version": legacy.get("signature_version", REVIEW_SIGNATURE_VERSION),
            },
            "signature_version": legacy.get("signature_version", REVIEW_SIGNATURE_VERSION),
        }

    def fingerprint(self) -> str:
        return _stable_fingerprint(self.model_dump(mode="json"))

    @property
    def project_id(self) -> str | None:
        return self.static_scope.project_id

    @property
    def task_id(self) -> str | None:
        return self.static_scope.task_id

    @property
    def platform(self) -> str | None:
        return self.static_scope.platform

    @property
    def root_id(self) -> str | None:
        return self.static_scope.root_id

    @property
    def language_family(self) -> str | None:
        return self.static_scope.language_family

    @property
    def ontology_fingerprint(self) -> str | None:
        return self.knowledge_state.ontology_fingerprint

    @property
    def overlay_fingerprint(self) -> str | None:
        return self.knowledge_state.overlay_fingerprint

    @property
    def reviewed_overlay_fingerprint(self) -> str | None:
        return self.knowledge_state.reviewed_overlay_fingerprint


def build_review_context_index(
    mentions: list[MentionRecord],
    registry: OntologyRegistry,
    *,
    source_family: str | None = None,
    source_domain: str | None = None,
) -> dict[str, ReviewContext]:
    return {
        mention.mention_id: ReviewContext.from_mention(
            mention,
            registry,
            source_family=source_family,
            source_domain=source_domain,
        )
        for mention in mentions
    }
