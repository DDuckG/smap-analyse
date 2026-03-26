import json
from pydantic import BaseModel, Field

class PromotedSemanticKnowledgeRecord(BaseModel):
    record_id: str
    promoted_from_record_id: str
    semantic_kind: str
    mention_id: str
    source_uap_id: str | None = None
    segment_id: str | None = None
    evidence_span_signature: str | None = None
    source_target_key: str | None = None
    source_aspect: str | None = None
    source_issue_category: str | None = None
    source_evidence_mode: str | None = None
    source_severity: str | None = None
    target_key: str | None = None
    aspect: str | None = None
    issue_category: str | None = None
    evidence_mode: str | None = None
    severity: str | None = None
    evidence_text: str | None = None
    normalized_evidence_text: str
    semantic_region_key: str
    notes: str | None = None
    reviewer: str
    prediction_model_versions: list[str] = Field(default_factory=list)
    promoted_at: str = ''
    schema_version: str = 'promoted-semantic-knowledge-v1'

class PromotedTopicLineageRecord(BaseModel):
    record_id: str
    promoted_from_record_id: str
    source_topic_key: str
    canonical_topic_key: str
    canonical_topic_label: str
    reviewed_topic_id: str
    topic_lineage_id: str
    topic_signature: str | None = None
    topic_term_signature: str | None = None
    topic_profile_signature: str | None = None
    merge_into_topic_key: str | None = None
    usefulness_judgment: str | None = None
    stability_judgment: str | None = None
    notes: str | None = None
    reviewer: str
    promoted_at: str = ''
    schema_version: str = 'promoted-topic-lineage-v1'

class FeedbackStore:

    def __init__(self, settings):
        self.semantic_knowledge_path = settings.semantic_promoted_path
        self.topic_lineage_path = settings.topic_lineage_path
        settings.ensure_directories()

    def load_promoted_semantic_knowledge(self):
        return _load_records(self.semantic_knowledge_path, PromotedSemanticKnowledgeRecord)

    def load_topic_lineage(self):
        return _load_records(self.topic_lineage_path, PromotedTopicLineageRecord)

def _load_records(path, model_cls):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        records.append(model_cls.model_validate(json.loads(line)))
    return records
