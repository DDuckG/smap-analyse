from __future__ import annotations
from enum import StrEnum

class ReviewProblemClass(StrEnum):
    UNRESOLVED_ENTITY_CANDIDATE = 'unresolved_entity_candidate'
    AMBIGUOUS_ENTITY_MAPPING = 'ambiguous_entity_mapping'
    LOW_CONFIDENCE_CLASSIFICATION = 'low_confidence_classification'
    SUSPICIOUS_NOISE_CANDIDATE = 'suspicious_noise_candidate'
    ONTOLOGY_OR_OVERLAY_CONFLICT = 'ontology_or_overlay_conflict'

class ReviewStatus(StrEnum):
    PENDING = 'pending'
    GROUPED = 'grouped'
    ASSIGNED = 'assigned'
    IN_REVIEW = 'in_review'
    RESOLVED = 'resolved'
    REJECTED = 'rejected'
    APPLIED = 'applied'
    OBSOLETE = 'obsolete'

class ReviewResolutionScope(StrEnum):
    ITEM_ONLY = 'item_only'
    GROUP = 'group'
    FUTURE_OVERLAY = 'future_overlay'
    FUTURE_NOISE_SUPPRESSION = 'future_noise_suppression'
    CLOSE_ONLY = 'close_only'

class ReviewAction(StrEnum):
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'
    REMAPPED = 'remapped'
    MARKED_NOISE = 'marked_noise'
    KEPT_UNRESOLVED = 'kept_unresolved'

class ReviewScopeLevel(StrEnum):
    GLOBAL = 'global'
    PROJECT = 'project'
    TASK = 'task'
    PROJECT_ONTOLOGY = 'project_ontology'
    TASK_ONTOLOGY = 'task_ontology'
    THREAD_ONTOLOGY = 'thread_ontology'

class SemanticMatchMode(StrEnum):
    NORMALIZED_FORM = 'normalized_form'
    NORMALIZED_FORM_AND_TYPE = 'normalized_form_and_type'
    AMBIGUITY_SET = 'ambiguity_set'
    CLASSIFICATION_LABEL = 'classification_label'

class FutureEffectType(StrEnum):
    NONE = 'none'
    APPLY_ALIAS_OVERLAY = 'apply_alias_overlay'
    APPLY_NOISE_SUPPRESSION = 'apply_noise_suppression'

class KnowledgeMatchMode(StrEnum):
    IGNORE = 'ignore'
    BASE_ONTOLOGY = 'base_ontology'
    BASE_WITH_NON_REVIEW_OVERLAYS = 'base_with_non_review_overlays'
    REVIEWED_STATE_SENSITIVE = 'reviewed_state_sensitive'
    EXACT_FULL_STATE = 'exact_full_state'

class AuthorityLevel(StrEnum):
    GROUP = 'group'
    ITEM = 'item'

class ScopeRelationship(StrEnum):
    EXACT_SAME_SCOPE = 'exact_same_scope'
    BROADER_SCOPE = 'broader_scope'
    NARROWER_SCOPE = 'narrower_scope'
    OVERLAPPING_SCOPE = 'overlapping_scope'
    DISJOINT_SCOPE = 'disjoint_scope'
    INCOMPATIBLE_SCOPE = 'incompatible_scope'

class KnowledgeStateRelationship(StrEnum):
    SAME_KNOWLEDGE_STATE = 'same_knowledge_state'
    BROADER_KNOWLEDGE_COMPATIBILITY = 'broader_knowledge_compatibility'
    NARROWER_KNOWLEDGE_COMPATIBILITY = 'narrower_knowledge_compatibility'
    OVERLAPPING_KNOWLEDGE_STATE = 'overlapping_knowledge_state'
    INCOMPATIBLE_KNOWLEDGE_STATE = 'incompatible_knowledge_state'

class ContributionAction(StrEnum):
    IDENTICAL = 'identical'
    REPLACE_EXISTING = 'replace_existing'
    COEXIST = 'coexist'
    REJECT_CONFLICT = 'reject_conflict'
