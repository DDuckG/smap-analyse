from __future__ import annotations
import hashlib
import json
from collections import Counter, defaultdict
from typing import Literal
from smap.canonicalization.alias import normalize_alias
from smap.enrichers.models import EntityCandidateClusterFact, EntityFact, FactProvenance
from smap.normalization.models import MentionRecord

def cluster_unresolved_entity_facts(entity_facts, mentions):
    mention_by_id = {mention.mention_id: mention for mention in mentions}
    grouped = defaultdict(list)
    for fact in entity_facts:
        if fact.canonical_entity_id is not None or fact.concept_entity_id is not None:
            continue
        if fact.unresolved_reason in {'no_candidate_found', 'blocked_by_other_domain_catalog'}:
            continue
        normalized_surface = normalize_alias(fact.candidate_text)
        if not normalized_surface:
            continue
        grouped[normalized_surface, fact.entity_type].append(fact)
    cluster_records = []
    cluster_info = {}
    for (normalized_surface, entity_type), facts in grouped.items():
        unique_mentions = sorted({fact.mention_id for fact in facts})
        if len(unique_mentions) < 2:
            continue
        representative_surface = Counter((fact.candidate_text for fact in facts)).most_common(1)[0][0]
        payload = {'normalized_surface': normalized_surface, 'entity_type': entity_type}
        cluster_id = 'candclu.' + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()[:12]
        cluster_size = len(unique_mentions)
        cluster_info[normalized_surface, entity_type] = (cluster_id, cluster_size)
        languages = sorted({mention.language for mention_id in unique_mentions if (mention := mention_by_id.get(mention_id)) is not None and mention.language})
        discovered_by = sorted({method for fact in facts for method in fact.discovered_by})
        candidate_ids = sorted({candidate_id for fact in facts for candidate_id in fact.canonical_candidate_ids})
        promotion_state = 'promotable' if cluster_size >= 3 else 'reviewable'
        provenance_fact = facts[0]
        cluster_records.append(EntityCandidateClusterFact(cluster_id=cluster_id, normalized_surface=normalized_surface, representative_surface=representative_surface, entity_type_hint=entity_type, mention_count=cluster_size, source_languages=languages, discovered_by=discovered_by, representative_mention_ids=unique_mentions[:5], candidate_canonical_ids=candidate_ids, promotion_state=promotion_state, provenance=FactProvenance(source_uap_id=provenance_fact.source_uap_id, mention_id=provenance_fact.mention_id, provider_version=provenance_fact.provenance.provider_version, rule_version='batch-candidate-clustering-v1', evidence_text=representative_surface, evidence_span=provenance_fact.provenance.evidence_span)))
    updated_facts = []
    for fact in entity_facts:
        normalized_surface = normalize_alias(fact.candidate_text)
        cluster = cluster_info.get((normalized_surface, fact.entity_type))
        if cluster is None:
            updated_facts.append(fact)
            continue
        cluster_id, cluster_size = cluster
        updated_facts.append(fact.model_copy(update={'knowledge_layer': 'batch_local_candidate', 'unresolved_cluster_id': cluster_id, 'unresolved_cluster_size': cluster_size}))
    cluster_records.sort(key=lambda item: (-item.mention_count, item.normalized_surface))
    return (updated_facts, cluster_records)
