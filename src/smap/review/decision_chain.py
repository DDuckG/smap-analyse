from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from smap.review.applicability_engine import ApplicabilityEngine, GroupDecisionLike
from smap.review.policy import DecisionApplicabilityPolicy


def effective_future_decision(
    decisions: Iterable[GroupDecisionLike],
    *,
    engine: ApplicabilityEngine | None = None,
) -> GroupDecisionLike | None:
    applicability_engine = engine or ApplicabilityEngine()
    return applicability_engine.effective_future_decision(decisions)


def effective_future_policy(
    decisions: Iterable[GroupDecisionLike],
    *,
    engine: ApplicabilityEngine | None = None,
) -> DecisionApplicabilityPolicy | None:
    applicability_engine = engine or ApplicabilityEngine()
    return applicability_engine.effective_policy(decisions)


def group_decision_sort_key(decision: Any) -> tuple[object, int]:
    return (getattr(decision, "created_at", None), int(getattr(decision, "review_group_decision_id", 0)))
