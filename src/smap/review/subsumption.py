from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from smap.ontology.models import AliasContribution, NoiseTerm
from smap.review.applicability_engine import ApplicabilityEngine
from smap.review.policy import DecisionApplicabilityPolicy
from smap.review.types import ContributionAction, KnowledgeStateRelationship, ScopeRelationship


@dataclass(slots=True)
class ContributionEvaluation:
    action: ContributionAction
    reason: str
    scope_relationship: ScopeRelationship
    knowledge_relationship: KnowledgeStateRelationship


@dataclass(slots=True)
class ContributionReconciliationResult:
    action: ContributionAction
    reason: str
    retained_existing: list[AliasContribution | NoiseTerm]
    replaced_existing: list[AliasContribution | NoiseTerm]
    coexisting_existing: list[AliasContribution | NoiseTerm]
    conflicting_existing: list[AliasContribution | NoiseTerm]
    evaluations: list[tuple[AliasContribution | NoiseTerm, ContributionEvaluation]]


@dataclass(slots=True)
class ContributionClusterMember:
    contribution: AliasContribution | NoiseTerm
    policy: DecisionApplicabilityPolicy | None
    evaluation: ContributionEvaluation
    effect_key: tuple[str, str | None]
    domain_fingerprint: str


class ContributionSubsumptionEngine:
    def __init__(self, applicability_engine: ApplicabilityEngine | None = None) -> None:
        self.applicability_engine = applicability_engine or ApplicabilityEngine()

    def evaluate_alias_contribution(
        self,
        *,
        existing: AliasContribution | NoiseTerm,
        incoming_policy: DecisionApplicabilityPolicy,
        incoming_target: str | None,
        incoming_kind: str,
        allow_authoritative_replacement: bool,
    ) -> ContributionEvaluation:
        existing_policy = self.applicability_engine.policy_from_provenance(existing.provenance)
        if existing_policy is None:
            return ContributionEvaluation(
                action=ContributionAction.REJECT_CONFLICT,
                reason="rejected_missing_existing_policy",
                scope_relationship=ScopeRelationship.INCOMPATIBLE_SCOPE,
                knowledge_relationship=KnowledgeStateRelationship.INCOMPATIBLE_KNOWLEDGE_STATE,
            )

        scope_relationship = self.applicability_engine.scope_relationship(
            existing_policy.valid_scope_key.static_scope,
            incoming_policy.valid_scope_key.static_scope,
        )
        knowledge_relationship = self.applicability_engine.knowledge_state_relationship(
            existing_policy,
            incoming_policy,
        )

        same_effect = self._same_effect(existing, incoming_target, incoming_kind)
        exact_applicability = (
            scope_relationship == ScopeRelationship.EXACT_SAME_SCOPE
            and knowledge_relationship == KnowledgeStateRelationship.SAME_KNOWLEDGE_STATE
        )
        compatible_domains = (
            scope_relationship not in {ScopeRelationship.DISJOINT_SCOPE, ScopeRelationship.INCOMPATIBLE_SCOPE}
            and knowledge_relationship != KnowledgeStateRelationship.INCOMPATIBLE_KNOWLEDGE_STATE
        )

        if same_effect and exact_applicability:
            return ContributionEvaluation(
                action=ContributionAction.REPLACE_EXISTING,
                reason="replaced_same_scope_same_alias",
                scope_relationship=scope_relationship,
                knowledge_relationship=knowledge_relationship,
            )

        if not compatible_domains:
            reason = (
                "coexist_different_knowledge_state"
                if knowledge_relationship == KnowledgeStateRelationship.INCOMPATIBLE_KNOWLEDGE_STATE
                else "coexist_disjoint_scope"
            )
            return ContributionEvaluation(
                action=ContributionAction.COEXIST,
                reason=reason,
                scope_relationship=scope_relationship,
                knowledge_relationship=knowledge_relationship,
            )

        if same_effect:
            coexist_reason = {
                ScopeRelationship.BROADER_SCOPE: "coexist_broader_scope",
                ScopeRelationship.NARROWER_SCOPE: "coexist_narrower_scope",
                ScopeRelationship.OVERLAPPING_SCOPE: "coexist_overlapping_scope",
            }.get(scope_relationship, "coexist_same_target")
            if scope_relationship == ScopeRelationship.EXACT_SAME_SCOPE:
                coexist_reason = {
                    KnowledgeStateRelationship.BROADER_KNOWLEDGE_COMPATIBILITY: "coexist_broader_knowledge_state",
                    KnowledgeStateRelationship.NARROWER_KNOWLEDGE_COMPATIBILITY: "coexist_narrower_knowledge_state",
                    KnowledgeStateRelationship.OVERLAPPING_KNOWLEDGE_STATE: "coexist_overlapping_knowledge_state",
                }.get(knowledge_relationship, coexist_reason)
            return ContributionEvaluation(
                action=ContributionAction.COEXIST,
                reason=coexist_reason,
                scope_relationship=scope_relationship,
                knowledge_relationship=knowledge_relationship,
            )

        if exact_applicability and allow_authoritative_replacement:
            return ContributionEvaluation(
                action=ContributionAction.REPLACE_EXISTING,
                reason="replaced_same_scope_due_to_authority",
                scope_relationship=scope_relationship,
                knowledge_relationship=knowledge_relationship,
            )

        if scope_relationship in {
            ScopeRelationship.BROADER_SCOPE,
            ScopeRelationship.NARROWER_SCOPE,
            ScopeRelationship.OVERLAPPING_SCOPE,
        } or knowledge_relationship in {
            KnowledgeStateRelationship.BROADER_KNOWLEDGE_COMPATIBILITY,
            KnowledgeStateRelationship.NARROWER_KNOWLEDGE_COMPATIBILITY,
            KnowledgeStateRelationship.OVERLAPPING_KNOWLEDGE_STATE,
        }:
            coexist_reason = {
                ScopeRelationship.BROADER_SCOPE: "coexist_broader_scope",
                ScopeRelationship.NARROWER_SCOPE: "coexist_narrower_scope",
                ScopeRelationship.OVERLAPPING_SCOPE: "coexist_overlapping_scope",
            }.get(scope_relationship, "coexist_same_scope_different_knowledge_state")
            if scope_relationship == ScopeRelationship.EXACT_SAME_SCOPE:
                coexist_reason = {
                    KnowledgeStateRelationship.BROADER_KNOWLEDGE_COMPATIBILITY: "coexist_broader_knowledge_state",
                    KnowledgeStateRelationship.NARROWER_KNOWLEDGE_COMPATIBILITY: "coexist_narrower_knowledge_state",
                    KnowledgeStateRelationship.OVERLAPPING_KNOWLEDGE_STATE: "coexist_overlapping_knowledge_state",
                }.get(knowledge_relationship, coexist_reason)
            return ContributionEvaluation(
                action=ContributionAction.COEXIST,
                reason=coexist_reason,
                scope_relationship=scope_relationship,
                knowledge_relationship=knowledge_relationship,
            )

        conflict_reason = (
            "rejected_conflicting_target_same_scope"
            if exact_applicability
            else "rejected_conflicting_target_overlapping_scope"
        )
        return ContributionEvaluation(
            action=ContributionAction.REJECT_CONFLICT,
            reason=conflict_reason,
            scope_relationship=scope_relationship,
            knowledge_relationship=knowledge_relationship,
        )

    @staticmethod
    def _same_effect(
        existing: AliasContribution | NoiseTerm,
        incoming_target: str | None,
        incoming_kind: str,
    ) -> bool:
        if isinstance(existing, AliasContribution):
            return incoming_kind == "alias" and existing.canonical_entity_id == incoming_target
        return incoming_kind == "noise" and incoming_target is None

    @staticmethod
    def _reason_priority(reason: str) -> tuple[int, str]:
        priority = {
            "rejected_missing_existing_policy": 0,
            "rejected_conflicting_target_same_scope": 1,
            "replaced_same_scope_due_to_authority": 2,
            "exact_replace": 3,
            "coexist_different_knowledge_state": 4,
            "coexist_narrower_scope": 5,
            "coexist_broader_scope": 6,
            "coexist_overlapping_scope": 7,
            "coexist_same_target": 8,
            "coexist_disjoint_scope": 9,
            "new_contribution": 10,
        }
        return priority.get(reason, 50), reason

    def _region_fingerprint(self, policy: DecisionApplicabilityPolicy) -> str:
        payload: dict[str, Any] = {
            "scope_level": policy.scope_level.value,
            "static_scope": self.applicability_engine.static_scope_constraints(
                policy.valid_scope_key.static_scope
            ),
            "knowledge_constraints": self.applicability_engine.knowledge_constraints(policy),
            "valid_signature_version": policy.valid_signature_version,
        }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _sort_key(contribution: AliasContribution | NoiseTerm) -> tuple[str, str, str]:
        if isinstance(contribution, AliasContribution):
            return (
                "alias",
                contribution.canonical_entity_id,
                json.dumps(contribution.provenance, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            )
        return (
            "noise",
            contribution.reason,
            json.dumps(contribution.provenance, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        )

    def _build_cluster_members(
        self,
        *,
        existing_contributions: list[AliasContribution | NoiseTerm],
        incoming_policy: DecisionApplicabilityPolicy,
        incoming_target: str | None,
        incoming_kind: str,
        allow_authoritative_replacement: bool,
    ) -> list[ContributionClusterMember]:
        members: list[ContributionClusterMember] = []
        for existing in sorted(existing_contributions, key=self._sort_key):
            evaluation = self.evaluate_alias_contribution(
                existing=existing,
                incoming_policy=incoming_policy,
                incoming_target=incoming_target,
                incoming_kind=incoming_kind,
                allow_authoritative_replacement=allow_authoritative_replacement,
            )
            policy = self.applicability_engine.policy_from_provenance(existing.provenance)
            effect_key = (
                "alias",
                existing.canonical_entity_id,
            ) if isinstance(existing, AliasContribution) else ("noise", None)
            domain_fingerprint = self._region_fingerprint(policy) if policy is not None else "missing-policy"
            members.append(
                ContributionClusterMember(
                    contribution=existing,
                    policy=policy,
                    evaluation=evaluation,
                    effect_key=effect_key,
                    domain_fingerprint=domain_fingerprint,
                )
            )
        return members

    def _cluster_replace_members(
        self,
        members: list[ContributionClusterMember],
        incoming_effect_key: tuple[str, str | None],
    ) -> tuple[list[ContributionClusterMember], list[ContributionClusterMember], str]:
        exact_domain_members = [
            member
            for member in members
            if member.evaluation.scope_relationship == ScopeRelationship.EXACT_SAME_SCOPE
            and member.evaluation.knowledge_relationship == KnowledgeStateRelationship.SAME_KNOWLEDGE_STATE
        ]
        same_effect = [member for member in exact_domain_members if member.effect_key == incoming_effect_key]
        conflicting = [member for member in exact_domain_members if member.effect_key != incoming_effect_key]
        replaced = [*same_effect, *conflicting]
        retained = [member for member in members if member not in replaced]
        if conflicting:
            return replaced, retained, "replaced_same_scope_due_to_authority"
        return replaced, retained, "exact_replace"

    def _cluster_conflicts(
        self,
        members: list[ContributionClusterMember],
        incoming_effect_key: tuple[str, str | None],
        *,
        allow_authoritative_replacement: bool,
    ) -> list[ContributionClusterMember]:
        conflicts: list[ContributionClusterMember] = []
        for member in members:
            if member.policy is None:
                conflicts.append(member)
                continue
            exact_domain = (
                member.evaluation.scope_relationship == ScopeRelationship.EXACT_SAME_SCOPE
                and member.evaluation.knowledge_relationship == KnowledgeStateRelationship.SAME_KNOWLEDGE_STATE
            )
            if exact_domain and member.effect_key != incoming_effect_key and not allow_authoritative_replacement:
                conflicts.append(member)
        return conflicts

    def reconcile_contributions(
        self,
        *,
        existing_contributions: list[AliasContribution | NoiseTerm],
        incoming_policy: DecisionApplicabilityPolicy,
        incoming_target: str | None,
        incoming_kind: str,
        allow_authoritative_replacement: bool,
    ) -> ContributionReconciliationResult:
        incoming_effect_key = ("alias", incoming_target) if incoming_kind == "alias" else ("noise", None)
        members = self._build_cluster_members(
            existing_contributions=existing_contributions,
            incoming_policy=incoming_policy,
            incoming_target=incoming_target,
            incoming_kind=incoming_kind,
            allow_authoritative_replacement=allow_authoritative_replacement,
        )
        conflicts = self._cluster_conflicts(
            members,
            incoming_effect_key,
            allow_authoritative_replacement=allow_authoritative_replacement,
        )
        if conflicts:
            conflict_reasons = {
                member.evaluation.reason
                if member.policy is not None
                else "rejected_missing_existing_policy"
                for member in conflicts
            }
            evaluations = [
                (
                    member.contribution,
                    ContributionEvaluation(
                        action=(
                            ContributionAction.REJECT_CONFLICT
                            if member in conflicts
                            else ContributionAction.COEXIST
                        ),
                        reason=(
                            member.evaluation.reason
                            if member in conflicts
                            else (
                                "retained_due_to_cluster_conflict"
                                if member.evaluation.action == ContributionAction.REPLACE_EXISTING
                                else member.evaluation.reason
                            )
                        ),
                        scope_relationship=member.evaluation.scope_relationship,
                        knowledge_relationship=member.evaluation.knowledge_relationship,
                    ),
                )
                for member in members
            ]
            return ContributionReconciliationResult(
                action=ContributionAction.REJECT_CONFLICT,
                reason=sorted(conflict_reasons, key=self._reason_priority)[0],
                retained_existing=[member.contribution for member in members],
                replaced_existing=[],
                coexisting_existing=[member.contribution for member in members if member not in conflicts],
                conflicting_existing=[member.contribution for member in conflicts],
                evaluations=evaluations,
            )

        replaced_members, retained_members, replace_reason = self._cluster_replace_members(
            members,
            incoming_effect_key,
        )
        if replaced_members:
            evaluations = [
                (
                    member.contribution,
                    ContributionEvaluation(
                        action=(
                            ContributionAction.REPLACE_EXISTING
                            if member in replaced_members
                            else ContributionAction.COEXIST
                        ),
                        reason=(
                            "exact_replace"
                            if member in replaced_members and member.effect_key == incoming_effect_key
                            else (
                                "replaced_same_scope_due_to_authority"
                                if member in replaced_members
                                else member.evaluation.reason
                            )
                        ),
                        scope_relationship=member.evaluation.scope_relationship,
                        knowledge_relationship=member.evaluation.knowledge_relationship,
                    ),
                )
                for member in members
            ]
            return ContributionReconciliationResult(
                action=ContributionAction.REPLACE_EXISTING,
                reason=replace_reason,
                retained_existing=[member.contribution for member in retained_members],
                replaced_existing=[member.contribution for member in replaced_members],
                coexisting_existing=[member.contribution for member in retained_members],
                conflicting_existing=[],
                evaluations=evaluations,
            )

        evaluations = [(member.contribution, member.evaluation) for member in members]
        coexist_reason = "new_contribution"
        if members:
            coexist_reason = sorted(
                {member.evaluation.reason for member in members if member.evaluation.action == ContributionAction.COEXIST},
                key=self._reason_priority,
            )[0]
        return ContributionReconciliationResult(
            action=ContributionAction.COEXIST if existing_contributions else ContributionAction.IDENTICAL,
            reason=coexist_reason,
            retained_existing=[member.contribution for member in members],
            replaced_existing=[],
            coexisting_existing=[member.contribution for member in members],
            conflicting_existing=[],
            evaluations=evaluations,
        )
