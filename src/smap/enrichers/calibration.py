from __future__ import annotations
from smap.enrichers.semantic_models import ScoreComponent

def clamp_confidence(base, components):
    total = base + sum((component.value for component in components))
    return round(max(0.05, min(0.99, total)), 3)

def sentiment_confidence_components(*, explicit_target, inherited_target, support_count, mixed_evidence, contrastive, negated, uncertain):
    components = []
    if explicit_target:
        components.append(ScoreComponent(name='explicit_target', value=0.12, reason='explicit target anchor'))
    if inherited_target:
        components.append(ScoreComponent(name='inherited_target_penalty', value=-0.12, reason='target inferred from thread context'))
    if support_count > 1:
        components.append(ScoreComponent(name='multiple_support', value=min(0.14, 0.04 * (support_count - 1)), reason='multiple supporting cues'))
    if mixed_evidence:
        components.append(ScoreComponent(name='mixed_penalty', value=-0.1, reason='conflicting local evidence'))
    if contrastive:
        components.append(ScoreComponent(name='contrast_penalty', value=-0.04, reason='contrast boundary in segment'))
    if negated:
        components.append(ScoreComponent(name='negation_penalty', value=-0.03, reason='negation scope applied'))
    if uncertain:
        components.append(ScoreComponent(name='uncertainty_penalty', value=-0.1, reason='question, rumor, or uncertainty cue'))
    return components

def issue_confidence_components(*, explicit_target, inherited_target, corroboration_count, uncertain, escalation, direct_mode):
    components = sentiment_confidence_components(explicit_target=explicit_target, inherited_target=inherited_target, support_count=1, mixed_evidence=False, contrastive=False, negated=False, uncertain=uncertain)
    if corroboration_count > 1 and (not uncertain):
        components.append(ScoreComponent(name='thread_corroboration', value=min(0.16, 0.05 * (corroboration_count - 1)), reason='similar issue observed elsewhere in the thread'))
    if uncertain:
        components.append(ScoreComponent(name='uncertain_issue_penalty', value=-0.12, reason='uncertain or hearsay evidence lowers issue confidence'))
    if escalation:
        components.append(ScoreComponent(name='escalation_signal', value=0.08, reason='escalation-like evidence'))
    if direct_mode:
        components.append(ScoreComponent(name='direct_evidence', value=0.06, reason='direct complaint or observation'))
    return components
