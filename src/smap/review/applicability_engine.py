from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, cast

from smap.ontology.models import AliasContribution, NoiseTerm, OverlayProvenanceValue
from smap.review.authority import (
    AuthorityResolver,
    GroupAuthorityResolution,
    ItemAuthorityResolution,
    ReviewDecisionLike,
    ReviewGroupDecisionLike,
)
from smap.review.context import ReviewContext, ScopeKey, SemanticSignature, StaticScopeKey
from smap.review.policy import DecisionApplicabilityPolicy, ReviewPolicyEngine
from smap.review.types import (
    FutureEffectType,
    KnowledgeMatchMode,
    KnowledgeStateRelationship,
    ScopeRelationship,
    SemanticMatchMode,
)

GroupDecisionLike = ReviewGroupDecisionLike


class ApplicabilityEngine:
    def __init__(self, policy_engine: ReviewPolicyEngine | None = None) -> None:
        self.policy_engine = policy_engine or ReviewPolicyEngine()
        self.authority_resolver = AuthorityResolver(self.policy_from_payload)

    def policy_from_payload(self, payload: dict[str, Any] | None) -> DecisionApplicabilityPolicy | None:
        if payload is None:
            return None
        return DecisionApplicabilityPolicy.model_validate(payload)

    def policy_from_provenance(
        self,
        provenance: Mapping[str, OverlayProvenanceValue],
    ) -> DecisionApplicabilityPolicy | None:
        raw_value = provenance.get("applicability_policy_json")
        if not isinstance(raw_value, str) or not raw_value.strip():
            return None
        return DecisionApplicabilityPolicy.model_validate(cast(object, json.loads(raw_value)))

    def scope_key_to_provenance(self, scope_key: ScopeKey) -> dict[str, OverlayProvenanceValue]:
        return {
            "scope_key_json": scope_key.model_dump_json(),
            "scope_key_fingerprint": scope_key.fingerprint(),
            "scope_level": scope_key.scope_level.value,
            "signature_version": scope_key.signature_version,
            "static_scope_json": scope_key.static_scope.model_dump_json(),
            "knowledge_state_json": scope_key.knowledge_state.model_dump_json(),
            "knowledge_state_fingerprint": scope_key.knowledge_state.fingerprint(),
        }

    def applicability_to_provenance(
        self,
        policy: DecisionApplicabilityPolicy,
    ) -> dict[str, OverlayProvenanceValue]:
        return {
            **self.scope_key_to_provenance(policy.valid_scope_key),
            "applicability_policy_json": policy.model_dump_json(),
            "applicability_policy_fingerprint": policy.fingerprint(),
            "future_effect": policy.future_effect.value,
            "authority_level": policy.authority_level.value,
            "terminates_future_authority": policy.terminates_future_authority,
            "match_mode": policy.match_mode.value,
            "problem_class": policy.applies_to_problem_class.value,
            "semantic_match_value": policy.semantic_match_value,
            "semantic_fingerprint": policy.semantic_fingerprint,
            "valid_signature_version": policy.valid_signature_version,
            "origin_context_fingerprint": policy.origin_context_fingerprint,
            "origin_knowledge_state_json": policy.origin_knowledge_state.model_dump_json(),
            "origin_knowledge_state_fingerprint": policy.origin_knowledge_state.fingerprint(),
            "knowledge_match_mode": policy.knowledge_match_mode.value,
        }

    def scope_key_from_provenance(self, provenance: Mapping[str, OverlayProvenanceValue]) -> ScopeKey | None:
        raw_value = provenance.get("scope_key_json")
        if not isinstance(raw_value, str) or not raw_value.strip():
            return None
        return ScopeKey.model_validate(cast(object, json.loads(raw_value)))

    def static_scope_matches(self, scope: StaticScopeKey, context: ReviewContext) -> bool:
        candidate = context.static_scope_key()
        return self._constraints_match(
            self._non_null_constraints(scope.model_dump(mode="json")),
            candidate.model_dump(mode="json"),
        )

    def knowledge_state_matches(
        self,
        policy: DecisionApplicabilityPolicy,
        context: ReviewContext,
    ) -> bool:
        constraints = self._knowledge_constraints(policy)
        if not constraints:
            return True
        return self._constraints_match(constraints, context.knowledge_state().model_dump(mode="json"))

    def scope_matches_context(self, scope_key: ScopeKey, context: ReviewContext) -> bool:
        if scope_key.signature_version != context.signature_version:
            return False
        return self.static_scope_matches(scope_key.static_scope, context)

    def scope_relationship(self, left: StaticScopeKey, right: StaticScopeKey) -> ScopeRelationship:
        return cast(
            ScopeRelationship,
            self._relationship_from_constraints(
                self._non_null_constraints(left.model_dump(mode="json")),
                self._non_null_constraints(right.model_dump(mode="json")),
                same=ScopeRelationship.EXACT_SAME_SCOPE,
                broader=ScopeRelationship.BROADER_SCOPE,
                narrower=ScopeRelationship.NARROWER_SCOPE,
                overlapping=ScopeRelationship.OVERLAPPING_SCOPE,
                disjoint=ScopeRelationship.DISJOINT_SCOPE,
            ),
        )

    def knowledge_state_relationship(
        self,
        left_policy: DecisionApplicabilityPolicy,
        right_policy: DecisionApplicabilityPolicy,
    ) -> KnowledgeStateRelationship:
        return cast(
            KnowledgeStateRelationship,
            self._relationship_from_constraints(
                self._knowledge_constraints(left_policy),
                self._knowledge_constraints(right_policy),
                same=KnowledgeStateRelationship.SAME_KNOWLEDGE_STATE,
                broader=KnowledgeStateRelationship.BROADER_KNOWLEDGE_COMPATIBILITY,
                narrower=KnowledgeStateRelationship.NARROWER_KNOWLEDGE_COMPATIBILITY,
                overlapping=KnowledgeStateRelationship.OVERLAPPING_KNOWLEDGE_STATE,
                disjoint=KnowledgeStateRelationship.INCOMPATIBLE_KNOWLEDGE_STATE,
            ),
        )

    def scope_keys_overlap(self, left: ScopeKey, right: ScopeKey) -> bool:
        if left.signature_version != right.signature_version:
            return False
        scope_relation = self.scope_relationship(left.static_scope, right.static_scope)
        if scope_relation in {ScopeRelationship.DISJOINT_SCOPE, ScopeRelationship.INCOMPATIBLE_SCOPE}:
            return False
        return self._constraints_overlap(
            self._non_null_constraints(left.knowledge_state.model_dump(mode="json")),
            self._non_null_constraints(right.knowledge_state.model_dump(mode="json")),
        )

    def groupable(
        self,
        left_signature: SemanticSignature,
        left_context: ReviewContext,
        right_signature: SemanticSignature,
        right_context: ReviewContext,
    ) -> bool:
        left_policy = self.policy_engine.grouping_policy_for(left_signature)
        right_policy = self.policy_engine.grouping_policy_for(right_signature)
        if left_policy != right_policy:
            return False
        return self.policy_engine.review_group_fingerprint(
            left_signature,
            self.policy_engine.scope_key_for(
                left_context,
                left_policy.scope_level,
                grouping_policy=left_policy,
                signature=left_signature,
            ),
        ) == self.policy_engine.review_group_fingerprint(
            right_signature,
            self.policy_engine.scope_key_for(
                right_context,
                right_policy.scope_level,
                grouping_policy=right_policy,
                signature=right_signature,
            ),
        )

    def decision_applies(
        self,
        policy: DecisionApplicabilityPolicy,
        signature: SemanticSignature,
        context: ReviewContext,
    ) -> bool:
        if policy.future_effect == FutureEffectType.NONE:
            return False
        if policy.valid_signature_version != signature.signature_version:
            return False
        if policy.applies_to_problem_class != signature.problem_class:
            return False
        if not self.scope_matches_context(policy.valid_scope_key, context):
            return False
        if not self.knowledge_state_matches(policy, context):
            return False
        return self.semantic_matches(policy, signature)

    def contribution_applies(
        self,
        provenance: Mapping[str, OverlayProvenanceValue],
        context: ReviewContext | None,
    ) -> bool:
        if context is None:
            return False
        policy = self.policy_from_provenance(provenance)
        if policy is not None:
            return self.scope_matches_context(policy.valid_scope_key, context) and self.knowledge_state_matches(policy, context)
        scope_key = self.scope_key_from_provenance(provenance)
        if scope_key is None:
            return False
        return self.scope_matches_context(scope_key, context)

    def group_authority_resolution(
        self,
        decisions: Iterable[ReviewGroupDecisionLike],
    ) -> GroupAuthorityResolution:
        return self.authority_resolver.resolve_group_authority(list(decisions))

    def item_authority_resolution(
        self,
        *,
        group_decisions: Iterable[ReviewGroupDecisionLike],
        item_decisions: Iterable[ReviewDecisionLike],
    ) -> ItemAuthorityResolution:
        return self.authority_resolver.resolve_item_authority(
            group_decisions=list(group_decisions),
            item_decisions=list(item_decisions),
        )

    def effective_future_decision(
        self,
        decisions: Iterable[ReviewGroupDecisionLike],
    ) -> ReviewGroupDecisionLike | None:
        return self.group_authority_resolution(decisions).authoritative_decision

    def effective_policy(
        self,
        decisions: Iterable[ReviewGroupDecisionLike],
    ) -> DecisionApplicabilityPolicy | None:
        decision = self.effective_future_decision(decisions)
        if decision is None:
            return None
        return self.policy_from_payload(decision.applicability_policy_payload)

    def alias_contribution_applies(
        self,
        contribution: AliasContribution,
        context: ReviewContext | None,
    ) -> bool:
        return contribution.source != "review" or self.contribution_applies(contribution.provenance, context)

    def noise_term_applies(self, noise_term: NoiseTerm, context: ReviewContext | None) -> bool:
        return noise_term.source != "review" or self.contribution_applies(noise_term.provenance, context)

    def knowledge_constraints(
        self,
        policy: DecisionApplicabilityPolicy,
    ) -> dict[str, Any]:
        return self._knowledge_constraints(policy)

    def static_scope_constraints(self, scope: StaticScopeKey) -> dict[str, Any]:
        return self._non_null_constraints(scope.model_dump(mode="json"))

    def semantic_matches(
        self,
        policy: DecisionApplicabilityPolicy,
        signature: SemanticSignature,
    ) -> bool:
        candidate_value = signature.match_value(policy.match_mode)
        if candidate_value == policy.semantic_match_value:
            return True
        if policy.match_mode != SemanticMatchMode.NORMALIZED_FORM_AND_TYPE:
            return False
        policy_normalized, _, policy_type = policy.semantic_match_value.partition("::")
        candidate_normalized, _, candidate_type = candidate_value.partition("::")
        if policy_normalized != candidate_normalized:
            return False
        if policy_type in {"", "-"} or candidate_type in {"", "-"}:
            return True
        return policy_type == candidate_type

    def _knowledge_constraints(
        self,
        policy: DecisionApplicabilityPolicy,
    ) -> dict[str, Any]:
        origin = policy.origin_knowledge_state.model_dump(mode="json")
        if policy.knowledge_match_mode == KnowledgeMatchMode.IGNORE:
            return {}
        if policy.knowledge_match_mode == KnowledgeMatchMode.BASE_ONTOLOGY:
            return self._non_null_constraints(
                {
                    "ontology_fingerprint": origin.get("ontology_fingerprint"),
                }
            )
        if policy.knowledge_match_mode == KnowledgeMatchMode.BASE_WITH_NON_REVIEW_OVERLAYS:
            return self._non_null_constraints(
                {
                    "ontology_fingerprint": origin.get("ontology_fingerprint"),
                    "overlay_fingerprint": origin.get("overlay_fingerprint"),
                }
            )
        if policy.knowledge_match_mode == KnowledgeMatchMode.REVIEWED_STATE_SENSITIVE:
            constraints = {
                "ontology_fingerprint": origin.get("ontology_fingerprint"),
                "overlay_fingerprint": origin.get("overlay_fingerprint"),
            }
            reviewed_overlay_fingerprint = origin.get("reviewed_overlay_fingerprint")
            if reviewed_overlay_fingerprint not in {None, ""}:
                constraints["reviewed_overlay_fingerprint"] = reviewed_overlay_fingerprint
            return self._non_null_constraints(constraints)
        return self._non_null_constraints(origin)

    @staticmethod
    def _relationship_from_constraints(
        left: dict[str, Any],
        right: dict[str, Any],
        *,
        same: ScopeRelationship | KnowledgeStateRelationship,
        broader: ScopeRelationship | KnowledgeStateRelationship,
        narrower: ScopeRelationship | KnowledgeStateRelationship,
        overlapping: ScopeRelationship | KnowledgeStateRelationship,
        disjoint: ScopeRelationship | KnowledgeStateRelationship,
    ) -> ScopeRelationship | KnowledgeStateRelationship:
        if not left and not right:
            return same
        if not ApplicabilityEngine._constraints_overlap(left, right):
            return disjoint
        left_subset = ApplicabilityEngine._is_subset(left, right)
        right_subset = ApplicabilityEngine._is_subset(right, left)
        if left_subset and right_subset:
            return same
        if left_subset:
            return broader
        if right_subset:
            return narrower
        return overlapping

    @staticmethod
    def _non_null_constraints(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key != "signature_version" and value is not None
        }

    @staticmethod
    def _constraints_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return all(right.get(key) == left_value for key, left_value in left.items())

    @staticmethod
    def _constraints_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
        shared_keys = set(left) & set(right)
        return all(left[key] == right[key] for key in shared_keys)

    @staticmethod
    def _is_subset(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return all(right.get(key) == value for key, value in left.items())
