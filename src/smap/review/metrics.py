from __future__ import annotations
from pydantic import BaseModel

class ReviewQueueSummary(BaseModel):
    raw_reviewable_items: int = 0
    created_items: int = 0
    grouped_items: int = 0
    active_groups: int = 0
    suppressed_by_prior_decision: int = 0
    unresolved_entity_candidates: int = 0
    repeat_unresolved_count: int = 0
    entity_raw_reviewable_items: int = 0
    entity_created_items: int = 0
    entity_grouped_items: int = 0
    entity_suppressed_items: int = 0
    classification_raw_reviewable_items: int = 0
    classification_created_items: int = 0
    classification_grouped_items: int = 0
    classification_suppressed_items: int = 0

    @property
    def duplicate_compression_ratio(self):
        denominator = self.created_items if self.created_items else 1
        return round(self.raw_reviewable_items / denominator, 3)

    @property
    def groupable_review_ratio(self):
        if self.raw_reviewable_items == 0:
            return 0.0
        return round(self.grouped_items / self.raw_reviewable_items, 3)

    @property
    def group_coverage_ratio(self):
        if self.created_items == 0:
            return 0.0
        return round(self.grouped_items / self.created_items, 3)

    @property
    def unresolved_to_review_conversion_ratio(self):
        if self.unresolved_entity_candidates == 0:
            return 0.0
        return round(self.entity_created_items / self.unresolved_entity_candidates, 3)

    @property
    def entity_review_items_per_unit(self):
        denominator = self.entity_created_items if self.entity_created_items else 1
        return round(self.entity_raw_reviewable_items / denominator, 3)

    @property
    def classification_review_items_per_unit(self):
        denominator = self.classification_created_items if self.classification_created_items else 1
        return round(self.classification_raw_reviewable_items / denominator, 3)
