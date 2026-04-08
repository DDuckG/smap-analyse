from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from smap.review.authority_policy import AuthorityCandidate, AuthorityPrecedencePolicy
from smap.review.policy import DecisionApplicabilityPolicy
from smap.review.types import AuthorityLevel, FutureEffectType


class ReviewGroupDecisionLike(Protocol):
    review_group_decision_id: int
    applicability_policy_payload: dict[str, Any] | None
    supersedes_review_group_decision_id: int | None
    terminated_by_review_group_decision_id: int | None
    created_at: datetime


class ReviewDecisionLike(Protocol):
    review_decision_id: int
    applicability_policy_payload: dict[str, Any] | None
    supersedes_review_decision_id: int | None
    terminated_by_review_decision_id: int | None
    created_at: datetime


@dataclass(slots=True)
class GroupAuthorityResolution:
    authoritative_decision: ReviewGroupDecisionLike | None
    superseded_ids: set[int] = field(default_factory=set)
    terminated_ids: set[int] = field(default_factory=set)
    terminating_decision_id: int | None = None
    reason: str | None = None


@dataclass(slots=True)
class ItemAuthorityResolution:
    source_level: AuthorityLevel | None
    authoritative_group_decision: ReviewGroupDecisionLike | None = None
    authoritative_item_decision: ReviewDecisionLike | None = None
    group_resolution: GroupAuthorityResolution | None = None
    item_terminated_group_authority: bool = False
    reason: str | None = None


class AuthorityResolver:
    def __init__(
        self,
        policy_loader: Callable[[dict[str, Any] | None], DecisionApplicabilityPolicy | None],
    ) -> None:
        self.policy_loader = policy_loader
        self.precedence_policy = AuthorityPrecedencePolicy()

    def resolve_group_authority(
        self,
        decisions: list[ReviewGroupDecisionLike],
    ) -> GroupAuthorityResolution:
        if not decisions:
            return GroupAuthorityResolution(authoritative_decision=None)

        sorted_decisions = sorted(decisions, key=lambda item: (item.created_at, item.review_group_decision_id))
        future_decisions: dict[int, ReviewGroupDecisionLike] = {}
        superseded_ids: set[int] = set()
        terminated_ids: set[int] = set()
        terminating_decision_id: int | None = None

        for decision in sorted_decisions:
            policy = self.policy_loader(decision.applicability_policy_payload)
            if policy is None:
                continue
            decision_id = decision.review_group_decision_id
            if policy.future_effect != FutureEffectType.NONE:
                future_decisions[decision_id] = decision
                if decision.supersedes_review_group_decision_id is not None:
                    superseded_ids.add(decision.supersedes_review_group_decision_id)
                if decision.terminated_by_review_group_decision_id is not None:
                    terminated_ids.add(decision_id)
                    terminating_decision_id = decision.terminated_by_review_group_decision_id
                continue
            if not policy.terminates_future_authority:
                continue
            current = self._current_group_authority(future_decisions, superseded_ids, terminated_ids)
            if current is None:
                continue
            terminated_ids.add(current.review_group_decision_id)
            terminating_decision_id = decision_id

        return GroupAuthorityResolution(
            authoritative_decision=self._current_group_authority(future_decisions, superseded_ids, terminated_ids),
            superseded_ids=superseded_ids,
            terminated_ids=terminated_ids,
            terminating_decision_id=terminating_decision_id,
            reason=self._group_reason(
                future_decisions,
                superseded_ids,
                terminated_ids,
                terminating_decision_id,
            ),
        )

    def resolve_item_authority(
        self,
        *,
        group_decisions: list[ReviewGroupDecisionLike],
        item_decisions: list[ReviewDecisionLike],
    ) -> ItemAuthorityResolution:
        group_resolution = self.resolve_group_authority(group_decisions)
        authoritative_group = group_resolution.authoritative_decision
        authoritative_item = self._resolve_item_future_authority(item_decisions)
        latest_item_terminator = self._latest_item_terminator(item_decisions)

        if authoritative_item is not None:
            group_policy = None
            item_policy = None
            choice = None
            if authoritative_group is not None:
                group_policy = self.policy_loader(authoritative_group.applicability_policy_payload)
                item_policy = self.policy_loader(authoritative_item.applicability_policy_payload)
                if group_policy is not None and item_policy is not None:
                    choice = self.precedence_policy.choose(
                        group_candidate=AuthorityCandidate(
                            authority_level=AuthorityLevel.GROUP,
                            created_at=authoritative_group.created_at,
                            decision_id=authoritative_group.review_group_decision_id,
                            policy=group_policy,
                        ),
                        item_candidate=AuthorityCandidate(
                            authority_level=AuthorityLevel.ITEM,
                            created_at=authoritative_item.created_at,
                            decision_id=authoritative_item.review_decision_id,
                            policy=item_policy,
                        ),
                    )
                    if choice.winner == AuthorityLevel.GROUP:
                        return ItemAuthorityResolution(
                            source_level=AuthorityLevel.GROUP,
                            authoritative_group_decision=authoritative_group,
                            authoritative_item_decision=authoritative_item,
                            group_resolution=group_resolution,
                            reason=choice.reason,
                        )
            return ItemAuthorityResolution(
                source_level=AuthorityLevel.ITEM,
                authoritative_group_decision=authoritative_group,
                authoritative_item_decision=authoritative_item,
                group_resolution=group_resolution,
                reason=choice.reason if choice is not None else "authoritative_future_decision_active",
            )

        if authoritative_group is None:
            return ItemAuthorityResolution(
                source_level=AuthorityLevel.ITEM if authoritative_item is not None else None,
                authoritative_item_decision=authoritative_item,
                group_resolution=group_resolution,
                reason="future_authority_terminated" if latest_item_terminator is not None else None,
            )

        if latest_item_terminator is not None and latest_item_terminator.created_at >= authoritative_group.created_at:
            return ItemAuthorityResolution(
                source_level=None,
                authoritative_group_decision=authoritative_group,
                group_resolution=group_resolution,
                item_terminated_group_authority=True,
                reason="future_authority_terminated",
            )

        return ItemAuthorityResolution(
            source_level=AuthorityLevel.GROUP,
            authoritative_group_decision=authoritative_group,
            authoritative_item_decision=authoritative_item,
            group_resolution=group_resolution,
            reason=group_resolution.reason or "authoritative_future_decision_active",
        )

    @staticmethod
    def _group_reason(
        future_decisions: dict[int, ReviewGroupDecisionLike],
        superseded_ids: set[int],
        terminated_ids: set[int],
        terminating_decision_id: int | None,
    ) -> str | None:
        authoritative = AuthorityResolver._current_group_authority(future_decisions, superseded_ids, terminated_ids)
        if authoritative is None:
            return "future_authority_terminated" if terminating_decision_id is not None else None
        if authoritative.supersedes_review_group_decision_id is not None:
            return "later_authority_supersedes"
        return "authoritative_future_decision_active"

    @staticmethod
    def _current_group_authority(
        future_decisions: dict[int, ReviewGroupDecisionLike],
        superseded_ids: set[int],
        terminated_ids: set[int],
    ) -> ReviewGroupDecisionLike | None:
        active = [
            decision
            for decision_id, decision in future_decisions.items()
            if decision_id not in superseded_ids and decision_id not in terminated_ids
        ]
        if not active:
            return None
        return max(active, key=lambda item: (item.created_at, item.review_group_decision_id))

    def _resolve_item_future_authority(
        self,
        decisions: list[ReviewDecisionLike],
    ) -> ReviewDecisionLike | None:
        if not decisions:
            return None
        sorted_decisions = sorted(decisions, key=lambda item: (item.created_at, item.review_decision_id))
        future_decisions: dict[int, ReviewDecisionLike] = {}
        superseded_ids: set[int] = set()
        terminated_ids: set[int] = set()

        for decision in sorted_decisions:
            policy = self.policy_loader(decision.applicability_policy_payload)
            if policy is None:
                continue
            decision_id = decision.review_decision_id
            if policy.future_effect != FutureEffectType.NONE:
                future_decisions[decision_id] = decision
                if decision.supersedes_review_decision_id is not None:
                    superseded_ids.add(decision.supersedes_review_decision_id)
                if decision.terminated_by_review_decision_id is not None:
                    terminated_ids.add(decision_id)
                continue
            if not policy.terminates_future_authority:
                continue
            current = self._current_item_authority(future_decisions, superseded_ids, terminated_ids)
            if current is None:
                continue
            terminated_ids.add(current.review_decision_id)

        return self._current_item_authority(future_decisions, superseded_ids, terminated_ids)

    def _latest_item_terminator(
        self,
        decisions: list[ReviewDecisionLike],
    ) -> ReviewDecisionLike | None:
        terminating_decisions: list[ReviewDecisionLike] = []
        for decision in decisions:
            policy = self.policy_loader(decision.applicability_policy_payload)
            if policy is None or not policy.terminates_future_authority:
                continue
            terminating_decisions.append(decision)
        if not terminating_decisions:
            return None
        return max(terminating_decisions, key=lambda item: (item.created_at, item.review_decision_id))

    @staticmethod
    def _current_item_authority(
        future_decisions: dict[int, ReviewDecisionLike],
        superseded_ids: set[int],
        terminated_ids: set[int],
    ) -> ReviewDecisionLike | None:
        active = [
            decision
            for decision_id, decision in future_decisions.items()
            if decision_id not in superseded_ids and decision_id not in terminated_ids
        ]
        if not active:
            return None
        return max(active, key=lambda item: (item.created_at, item.review_decision_id))
