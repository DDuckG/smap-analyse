from __future__ import annotations
import hashlib
import json
from typing import Any
from pydantic import BaseModel
from smap.core.settings import Settings
from smap.ontology.models import OntologyRegistry
from smap.review.context import REVIEW_SIGNATURE_VERSION, KnowledgeStateFingerprint, ReviewContext, ScopeKey, SemanticSignature, StaticScopeKey
from smap.review.relevant_knowledge import relevant_reviewed_knowledge_fingerprint
from smap.review.types import AuthorityLevel, FutureEffectType, KnowledgeMatchMode, ReviewProblemClass, ReviewResolutionScope, ReviewScopeLevel, SemanticMatchMode

def _policy_fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()[:16]

class ReviewGroupingPolicy(BaseModel):
    problem_class: ReviewProblemClass
    scope_level: ReviewScopeLevel
    match_mode: SemanticMatchMode
    knowledge_match_mode: KnowledgeMatchMode = KnowledgeMatchMode.BASE_WITH_NON_REVIEW_OVERLAYS
    signature_version: int = REVIEW_SIGNATURE_VERSION

class DecisionApplicabilityPolicy(BaseModel):
    applies_to_problem_class: ReviewProblemClass
    scope_level: ReviewScopeLevel
    authority_level: AuthorityLevel
    future_effect: FutureEffectType
    terminates_future_authority: bool = False
    match_mode: SemanticMatchMode
    knowledge_match_mode: KnowledgeMatchMode
    valid_signature_version: int
    valid_scope_key: ScopeKey
    semantic_match_value: str
    semantic_fingerprint: str
    origin_context_fingerprint: str
    origin_knowledge_state: KnowledgeStateFingerprint

    def fingerprint(self):
        return _policy_fingerprint(self.model_dump(mode='json'))

class ReviewPolicyEngine:

    def __init__(self, settings=None):
        self.settings = settings
        self._ontology_registry: OntologyRegistry | None = None
        self._ontology_registry_loaded = False

    def grouping_policy_for(self, signature):
        if signature.problem_class == ReviewProblemClass.LOW_CONFIDENCE_CLASSIFICATION:
            return ReviewGroupingPolicy(problem_class=signature.problem_class, scope_level=ReviewScopeLevel.TASK_ONTOLOGY, match_mode=SemanticMatchMode.CLASSIFICATION_LABEL, knowledge_match_mode=KnowledgeMatchMode.BASE_WITH_NON_REVIEW_OVERLAYS)
        if signature.problem_class == ReviewProblemClass.AMBIGUOUS_ENTITY_MAPPING:
            return ReviewGroupingPolicy(problem_class=signature.problem_class, scope_level=ReviewScopeLevel.PROJECT_ONTOLOGY, match_mode=SemanticMatchMode.AMBIGUITY_SET, knowledge_match_mode=KnowledgeMatchMode.REVIEWED_STATE_SENSITIVE)
        if signature.problem_class == ReviewProblemClass.SUSPICIOUS_NOISE_CANDIDATE:
            return ReviewGroupingPolicy(problem_class=signature.problem_class, scope_level=ReviewScopeLevel.PROJECT_ONTOLOGY, match_mode=SemanticMatchMode.NORMALIZED_FORM_AND_TYPE, knowledge_match_mode=KnowledgeMatchMode.REVIEWED_STATE_SENSITIVE)
        return ReviewGroupingPolicy(problem_class=signature.problem_class, scope_level=ReviewScopeLevel.PROJECT_ONTOLOGY, match_mode=SemanticMatchMode.NORMALIZED_FORM_AND_TYPE, knowledge_match_mode=KnowledgeMatchMode.REVIEWED_STATE_SENSITIVE)

    def scope_key_for(self, context, scope_level, *, grouping_policy=None, signature=None):
        static_scope = StaticScopeKey(signature_version=context.signature_version)
        knowledge_state = KnowledgeStateFingerprint(signature_version=context.signature_version)
        knowledge_match_mode = grouping_policy.knowledge_match_mode if grouping_policy is not None else KnowledgeMatchMode.BASE_WITH_NON_REVIEW_OVERLAYS
        if scope_level in {ReviewScopeLevel.PROJECT, ReviewScopeLevel.PROJECT_ONTOLOGY, ReviewScopeLevel.TASK, ReviewScopeLevel.TASK_ONTOLOGY, ReviewScopeLevel.THREAD_ONTOLOGY}:
            static_scope.project_id = context.project_id
        if scope_level in {ReviewScopeLevel.TASK, ReviewScopeLevel.TASK_ONTOLOGY, ReviewScopeLevel.THREAD_ONTOLOGY}:
            static_scope.task_id = context.task_id
        if scope_level == ReviewScopeLevel.THREAD_ONTOLOGY:
            static_scope.root_id = context.root_id
        if scope_level in {ReviewScopeLevel.PROJECT_ONTOLOGY, ReviewScopeLevel.TASK_ONTOLOGY, ReviewScopeLevel.THREAD_ONTOLOGY}:
            if knowledge_match_mode in {KnowledgeMatchMode.BASE_ONTOLOGY, KnowledgeMatchMode.BASE_WITH_NON_REVIEW_OVERLAYS, KnowledgeMatchMode.REVIEWED_STATE_SENSITIVE, KnowledgeMatchMode.EXACT_FULL_STATE}:
                knowledge_state.ontology_fingerprint = context.ontology_fingerprint
            if knowledge_match_mode in {KnowledgeMatchMode.BASE_WITH_NON_REVIEW_OVERLAYS, KnowledgeMatchMode.REVIEWED_STATE_SENSITIVE, KnowledgeMatchMode.EXACT_FULL_STATE}:
                knowledge_state.overlay_fingerprint = context.overlay_fingerprint
            if knowledge_match_mode in {KnowledgeMatchMode.REVIEWED_STATE_SENSITIVE, KnowledgeMatchMode.EXACT_FULL_STATE}:
                knowledge_state.reviewed_overlay_fingerprint = self._grouping_reviewed_knowledge_fingerprint(context=context, signature=signature) if grouping_policy is not None else context.reviewed_overlay_fingerprint
        return ScopeKey(scope_level=scope_level, static_scope=static_scope, knowledge_state=knowledge_state, signature_version=context.signature_version)

    def _grouping_reviewed_knowledge_fingerprint(self, *, context, signature):
        if signature is None:
            return 'none'
        registry = self._get_ontology_registry()
        if registry is None:
            return 'none'
        return relevant_reviewed_knowledge_fingerprint(registry, signature=signature, context=context)

    def _get_ontology_registry(self):
        if self._ontology_registry_loaded:
            return self._ontology_registry
        self._ontology_registry_loaded = True
        if self.settings is None:
            return None
        from smap.ontology.runtime import load_runtime_ontology
        self._ontology_registry = load_runtime_ontology(self.settings).registry
        return self._ontology_registry

    def review_group_fingerprint(self, signature, scope_key):
        return _policy_fingerprint({'signature_version': signature.signature_version, 'problem_class': signature.problem_class.value, 'semantic_fingerprint': signature.fingerprint(), 'scope_key_fingerprint': scope_key.fingerprint()})

    def build_applicability_policy(self, signature, context, scope_key, *, resolution_scope, authority_level, knowledge_match_mode=None, terminate_future_authority=None):
        grouping_policy = self.grouping_policy_for(signature)
        if resolution_scope == ReviewResolutionScope.FUTURE_OVERLAY.value:
            future_effect = FutureEffectType.APPLY_ALIAS_OVERLAY
        elif resolution_scope == ReviewResolutionScope.FUTURE_NOISE_SUPPRESSION.value:
            future_effect = FutureEffectType.APPLY_NOISE_SUPPRESSION
        else:
            future_effect = FutureEffectType.NONE
        if knowledge_match_mode is None:
            if scope_key.knowledge_state.ontology_fingerprint is None:
                effective_knowledge_match_mode = KnowledgeMatchMode.IGNORE
            elif future_effect == FutureEffectType.NONE:
                effective_knowledge_match_mode = KnowledgeMatchMode.EXACT_FULL_STATE
            else:
                effective_knowledge_match_mode = KnowledgeMatchMode.BASE_WITH_NON_REVIEW_OVERLAYS
        else:
            effective_knowledge_match_mode = knowledge_match_mode
        if terminate_future_authority is None:
            effective_terminate_future_authority = future_effect == FutureEffectType.NONE and (resolution_scope == ReviewResolutionScope.CLOSE_ONLY.value or authority_level == AuthorityLevel.GROUP)
        else:
            effective_terminate_future_authority = terminate_future_authority
        return DecisionApplicabilityPolicy(applies_to_problem_class=signature.problem_class, scope_level=scope_key.scope_level, authority_level=authority_level, future_effect=future_effect, terminates_future_authority=effective_terminate_future_authority, match_mode=grouping_policy.match_mode, knowledge_match_mode=effective_knowledge_match_mode, valid_signature_version=signature.signature_version, valid_scope_key=scope_key, semantic_match_value=signature.match_value(grouping_policy.match_mode), semantic_fingerprint=signature.fingerprint(), origin_context_fingerprint=context.fingerprint(), origin_knowledge_state=context.knowledge_state())
