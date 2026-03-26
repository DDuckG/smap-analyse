from __future__ import annotations
from smap.canonicalization.alias import normalize_alias
from smap.enrichers.models import TopicArtifactFact
from smap.hil.feedback_store import PromotedTopicLineageRecord

def apply_topic_lineage(artifact, records):
    match, resolution = resolve_topic_lineage(artifact, records)
    if match is None:
        return artifact.model_copy(update={'effective_topic_key': artifact.effective_topic_key or artifact.topic_key, 'effective_topic_label': artifact.effective_topic_label or artifact.topic_label, 'reviewed_topic_id': artifact.reviewed_topic_id or artifact.topic_key, 'topic_lineage_id': artifact.topic_lineage_id or default_topic_lineage_id(artifact), 'topic_review_resolution': resolution})
    effective_topic_key = match.canonical_topic_key or match.reviewed_topic_id or artifact.topic_key
    effective_topic_label = match.canonical_topic_label or artifact.reviewed_label_override or artifact.topic_label
    return artifact.model_copy(update={'effective_topic_key': effective_topic_key, 'effective_topic_label': effective_topic_label, 'reviewed_topic_id': match.reviewed_topic_id, 'topic_lineage_id': match.topic_lineage_id, 'reviewed_merge_target_key': match.merge_into_topic_key, 'reviewed_label_override': match.canonical_topic_label, 'reviewed_usefulness_judgment': match.usefulness_judgment, 'reviewed_stability_judgment': match.stability_judgment, 'topic_review_resolution': resolution})

def default_topic_lineage_id(artifact):
    base = artifact.effective_topic_key or artifact.topic_profile_signature or artifact.topic_signature or artifact.topic_term_signature or artifact.topic_key
    return normalize_alias(base) or base

def topic_lineage_matches(current, previous):
    current_effective_key = current.effective_topic_key or current.reviewed_topic_id or current.topic_key
    previous_effective_key = previous.effective_topic_key or previous.reviewed_topic_id or previous.topic_key
    if current_effective_key and previous_effective_key and (current_effective_key == previous_effective_key):
        return True
    current_id = current.topic_lineage_id or default_topic_lineage_id(current)
    previous_id = previous.topic_lineage_id or default_topic_lineage_id(previous)
    return bool(current_id and previous_id and (current_id == previous_id))

def resolve_topic_lineage(artifact, records):
    matches = _matching_records(records, topic_profile_signature=artifact.topic_profile_signature, topic_signature=artifact.topic_signature, topic_term_signature=artifact.topic_term_signature, topic_key=artifact.topic_key)
    if not matches:
        return (None, 'raw_identity')
    highest_priority = max((priority for priority, _ in matches))
    prioritized = [record for priority, record in matches if priority == highest_priority]
    resolved_identities = {(record.canonical_topic_key, record.reviewed_topic_id, record.topic_lineage_id) for record in prioritized}
    if len(resolved_identities) > 1:
        return (None, 'conflicting_reviewed_lineage')
    winning = sorted(prioritized, key=lambda item: (item.promoted_at, item.record_id), reverse=True)[0]
    match_kind = {4: 'topic_profile_signature', 3: 'topic_signature', 2: 'topic_term_signature', 1: 'topic_key'}.get(highest_priority, 'raw')
    if winning.merge_into_topic_key:
        return (winning, f'reviewed_merge:{match_kind}')
    return (winning, f'reviewed_label:{match_kind}')

def _matching_records(records, *, topic_profile_signature, topic_signature, topic_term_signature, topic_key):
    if not records:
        return []
    matches = []
    for record in records:
        if topic_profile_signature and record.topic_profile_signature == topic_profile_signature:
            matches.append((4, record))
            continue
        if topic_signature and record.topic_signature == topic_signature:
            matches.append((3, record))
            continue
        if topic_term_signature and record.topic_term_signature == topic_term_signature:
            matches.append((2, record))
            continue
        if record.source_topic_key == topic_key or record.canonical_topic_key == topic_key:
            matches.append((1, record))
    return matches
