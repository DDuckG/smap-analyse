from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from smap.review.policy import DecisionApplicabilityPolicy
from smap.review.types import AuthorityLevel


@dataclass(slots=True)
class AuthorityCandidate:
    authority_level: AuthorityLevel
    created_at: datetime
    decision_id: int
    policy: DecisionApplicabilityPolicy


@dataclass(slots=True)
class AuthoritySelection:
    winner: AuthorityLevel
    reason: str


class AuthorityPrecedencePolicy:
    def choose(
        self,
        *,
        group_candidate: AuthorityCandidate,
        item_candidate: AuthorityCandidate,
    ) -> AuthoritySelection:
        group_scope_rank = self._scope_specificity_rank(group_candidate.policy)
        item_scope_rank = self._scope_specificity_rank(item_candidate.policy)
        if item_scope_rank > group_scope_rank:
            return AuthoritySelection(winner=AuthorityLevel.ITEM, reason="narrower_scope_override")
        if group_scope_rank > item_scope_rank:
            return AuthoritySelection(winner=AuthorityLevel.GROUP, reason="broader_authority_retained")

        group_knowledge_rank = self._knowledge_specificity_rank(group_candidate.policy)
        item_knowledge_rank = self._knowledge_specificity_rank(item_candidate.policy)
        if item_knowledge_rank > group_knowledge_rank:
            return AuthoritySelection(winner=AuthorityLevel.ITEM, reason="narrower_knowledge_state_override")
        if group_knowledge_rank > item_knowledge_rank:
            return AuthoritySelection(winner=AuthorityLevel.GROUP, reason="broader_authority_retained")

        group_match_rank = self._match_specificity_rank(group_candidate.policy)
        item_match_rank = self._match_specificity_rank(item_candidate.policy)
        if item_match_rank > group_match_rank:
            return AuthoritySelection(winner=AuthorityLevel.ITEM, reason="exact_match_override")
        if group_match_rank > item_match_rank:
            return AuthoritySelection(winner=AuthorityLevel.GROUP, reason="broader_authority_retained")

        if item_candidate.authority_level != group_candidate.authority_level:
            return AuthoritySelection(winner=AuthorityLevel.ITEM, reason="item_precedence_over_group")

        if (
            item_candidate.created_at,
            item_candidate.decision_id,
        ) >= (
            group_candidate.created_at,
            group_candidate.decision_id,
        ):
            return AuthoritySelection(winner=AuthorityLevel.ITEM, reason="later_authority_supersedes")
        return AuthoritySelection(winner=AuthorityLevel.GROUP, reason="broader_authority_retained")

    @staticmethod
    def _scope_specificity_rank(policy: DecisionApplicabilityPolicy) -> int:
        return sum(
            1
            for key, value in policy.valid_scope_key.static_scope.model_dump(mode="json").items()
            if key != "signature_version" and value is not None
        )

    @staticmethod
    def _knowledge_specificity_rank(policy: DecisionApplicabilityPolicy) -> int:
        order = {
            "ignore": 0,
            "base_ontology": 1,
            "base_with_non_review_overlays": 2,
            "reviewed_state_sensitive": 3,
            "exact_full_state": 4,
        }
        return order[policy.knowledge_match_mode.value]

    @staticmethod
    def _match_specificity_rank(policy: DecisionApplicabilityPolicy) -> int:
        order = {
            "normalized_form": 0,
            "classification_label": 1,
            "normalized_form_and_type": 2,
            "ambiguity_set": 3,
        }
        return order[policy.match_mode.value]
