from __future__ import annotations
import re
from collections import Counter
from smap.canonicalization.alias import normalize_alias
from smap.providers.base import LanguageIdentificationResult, LanguageIdProvider, ProviderProvenance
_TOKEN_RE = re.compile("[\\w']+", flags=re.UNICODE)
_LATIN_LANGUAGE_HINTS = {'en': {'and', 'are', 'bad', 'but', 'charging', 'for', 'good', 'link', 'not', 'review', 'slow', 'that', 'the', 'this', 'with'}, 'vi': {'ban', 'bác', 'cho', 'của', 'cua', 'đẹp', 'dep', 'khong', 'la', 'may', 'nha', 'nhe', 'thoi', 'vay', 'voi', 'xe', 'vậy', 'nhé', 'thôi', 'máy', 'không', 'mà'}, 'es': {'con', 'de', 'el', 'esta', 'muy', 'para', 'pero', 'que', 'una', 'y'}, 'fr': {'avec', 'ce', 'dans', 'des', 'est', 'mais', 'pas', 'pour', 'que', 'une'}, 'de': {'aber', 'das', 'dem', 'die', 'gut', 'mit', 'nicht', 'und'}, 'pt': {'com', 'esta', 'mais', 'muito', 'nao', 'para', 'por', 'que', 'uma', 'voce'}, 'id': {'banget', 'dan', 'dengan', 'ini', 'itu', 'tidak', 'untuk', 'yang'}}
_VI_MARKERS = {'đ', 'ă', 'â', 'ê', 'ô', 'ơ', 'ư'}

def canonicalize_language_label(value):
    if value is None:
        return 'unknown'
    cleaned = value.strip().lower()
    if cleaned.startswith('__label__'):
        cleaned = cleaned.removeprefix('__label__')
    cleaned = cleaned.replace('_', '-')
    primary = cleaned.split('-', 1)[0]
    if primary == 'jp':
        return 'ja'
    if primary == 'in':
        return 'id'
    if re.fullmatch('[a-z]{2,3}', primary):
        return primary
    return cleaned or 'unknown'

def detect_script_languages(text):
    counts = Counter()
    for char in text:
        codepoint = ord(char)
        if 12352 <= codepoint <= 12543:
            counts['ja'] += 1
        elif 19968 <= codepoint <= 40959:
            counts['zh'] += 1
        elif 44032 <= codepoint <= 55215:
            counts['ko'] += 1
        elif 1024 <= codepoint <= 1279:
            counts['ru'] += 1
        elif 1536 <= codepoint <= 1791:
            counts['ar'] += 1
        elif 2304 <= codepoint <= 2431:
            counts['hi'] += 1
        elif 3584 <= codepoint <= 3711:
            counts['th'] += 1
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {language: round(count / total, 4) for language, count in counts.items()}

def looks_mixed_language(text, candidate_scores, *, mixed_confidence_threshold, mixed_gap_threshold, script_override_enabled):
    ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) >= 2:
        top_language, top_score = ranked[0]
        second_language, second_score = ranked[1]
        if top_language != second_language and (second_score >= mixed_confidence_threshold * 0.45 and top_score - second_score <= mixed_gap_threshold or ({top_language, second_language} == {'vi', 'en'} and second_score >= 0.22 and (top_score - second_score <= 0.24))):
            return True
    if not script_override_enabled:
        return False
    script_scores = detect_script_languages(text)
    return len(script_scores) >= 2 and max(script_scores.values(), default=0.0) < 0.88

def heuristic_language_candidates(text):
    normalized = normalize_alias(text)
    tokens = [token for token in _TOKEN_RE.findall(normalized) if token]
    if not tokens:
        script_scores = detect_script_languages(text)
        return script_scores or {'unknown': 0.0}
    scores = {}
    token_count = max(len(tokens), 1)
    for language, hints in _LATIN_LANGUAGE_HINTS.items():
        overlap = sum((1 for token in tokens if token in hints))
        if overlap <= 0:
            continue
        coverage = overlap / token_count
        bonus = 0.12 if language == 'vi' and any((ord(char) > 127 for char in text)) else 0.0
        scores[language] = round(min(coverage + bonus, 0.96), 4)
    for language, score in detect_script_languages(text).items():
        scores[language] = max(scores.get(language, 0.0), round(min(0.55 + score * 0.4, 0.98), 4))
    if any((marker in text.casefold() for marker in _VI_MARKERS)):
        scores['vi'] = max(scores.get('vi', 0.0), 0.52)
    if not scores:
        return {'unknown': 0.0}
    return scores

def best_language_candidate(candidates, *, text, mixed_confidence_threshold, mixed_gap_threshold, script_override_enabled):
    filtered = {language: score for language, score in candidates.items() if language and language != 'unknown'}
    if not filtered:
        return ('unknown', 0.0, candidates)
    ranked = sorted(filtered.items(), key=lambda item: (-item[1], item[0]))
    if looks_mixed_language(text, filtered, mixed_confidence_threshold=mixed_confidence_threshold, mixed_gap_threshold=mixed_gap_threshold, script_override_enabled=script_override_enabled):
        top_score = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else top_score
        return ('mixed', round(max((top_score + second_score) / 2.0, mixed_confidence_threshold), 4), filtered)
    return (ranked[0][0], ranked[0][1], filtered)

class HeuristicLanguageIdProvider(LanguageIdProvider):

    def __init__(self, *, mixed_confidence_threshold=0.55, mixed_gap_threshold=0.12, script_override_enabled=True):
        self.version = 'heuristic-lid-v2'
        self.mixed_confidence_threshold = mixed_confidence_threshold
        self.mixed_gap_threshold = mixed_gap_threshold
        self.script_override_enabled = script_override_enabled
        self.provenance = ProviderProvenance(provider_kind='language_id', provider_name='heuristic_lid', provider_version=self.version, model_id='heuristic-social-lid', device='cpu', run_metadata={'mixed_confidence_threshold': mixed_confidence_threshold, 'mixed_gap_threshold': mixed_gap_threshold, 'script_override_enabled': script_override_enabled})

    def detect(self, text):
        candidates = heuristic_language_candidates(text)
        language, confidence, filtered = best_language_candidate(candidates, text=text, mixed_confidence_threshold=self.mixed_confidence_threshold, mixed_gap_threshold=self.mixed_gap_threshold, script_override_enabled=self.script_override_enabled)
        top_candidates = sorted(filtered.items(), key=lambda item: (-item[1], item[0]))[:3]
        return LanguageIdentificationResult(language=language, confidence=confidence, provider_provenance=self.provenance, metadata={'candidate_count': len(filtered), 'top_candidates': ', '.join((f'{item[0]}:{item[1]:.3f}' for item in top_candidates))})
