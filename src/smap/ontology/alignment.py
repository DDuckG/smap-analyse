from __future__ import annotations
from smap.enrichers.models import EnrichmentBundle
from smap.ontology.models import OntologyRegistry
PROVISIONAL_PREFIX = 'provisional.'

def _is_allowed(label, allowed):
    return label in allowed or label.startswith(PROVISIONAL_PREFIX)

def collect_alignment_errors(enrichment, ontology):
    errors = []
    for entity_fact in enrichment.entity_facts:
        if entity_fact.entity_type and entity_fact.entity_type not in ontology.entity_types and (not entity_fact.entity_type.startswith(PROVISIONAL_PREFIX)):
            errors.append(f'unknown entity_type: {entity_fact.entity_type}')
    for intent_fact in enrichment.intent_facts:
        if not _is_allowed(intent_fact.intent, ontology.intent_category_ids):
            errors.append(f'unknown intent: {intent_fact.intent}')
    for aspect_fact in enrichment.aspect_opinion_facts:
        if not _is_allowed(aspect_fact.aspect, ontology.aspect_category_ids):
            errors.append(f'unknown aspect: {aspect_fact.aspect}')
    for issue_fact in enrichment.issue_signal_facts:
        if not _is_allowed(issue_fact.issue_category, ontology.issue_category_ids):
            errors.append(f'unknown issue_category: {issue_fact.issue_category}')
    for source_fact in enrichment.source_influence_facts:
        if not _is_allowed(source_fact.channel, ontology.source_channel_ids):
            errors.append(f'unknown source channel: {source_fact.channel}')
    return errors
