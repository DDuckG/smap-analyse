from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from smap.providers.base import LanguageIdentificationResult, LanguageIdProvider, LanguageSource, ProviderProvenance
from smap.providers.lid_heuristic import HeuristicLanguageIdProvider, canonicalize_language_label
URL_RE = re.compile('https?://\\S+')
MENTION_RE = re.compile('(?<!\\w)@[\\w\\d_.]+', re.UNICODE)
HASHTAG_RE = re.compile('#([\\w\\d_À-ỹà-ỹ-]+)', re.UNICODE)
EMOJI_RE = re.compile('[🌀-🗿😀-🙏🚀-\U0001f6ff🜀-🝿🤀-🧿🩰-\U0001faff]+', flags=re.UNICODE)
WHITESPACE_RE = re.compile('\\s+')
REPEATED_ALPHA_RE = re.compile('([A-Za-zÀ-ỹà-ỹ])\\1{2,}', re.UNICODE)
REPEATED_PUNCT_RE = re.compile('([!?.,])\\1{1,}')
PUNCT_COMPRESS_RE = re.compile('[^\\w\\s#:/+-]', flags=re.UNICODE)
TOKEN_RE = re.compile('[A-Za-zÀ-ỹà-ỹ0-9][A-Za-zÀ-ỹà-ỹ0-9+-]*', flags=re.UNICODE)
_HEURISTIC_LANGUAGE_PROVIDER = HeuristicLanguageIdProvider()
_VI_STOPWORDS = {'ai', 'anh', 'bác', 'ban', 'bạn', 'cham', 'chua', 'cho', 'co', 'của', 'cua', 'da', 'được', 'giao', 'khong', 'không', 'là', 'la', 'mấy', 'mình', 'này', 'nha', 'nhé', 'nhanh', 'nhung', 'nay', 'roi', 'rồi', 'sao', 'tang', 'thu', 'thế', 'thì', 'và', 'vậy', 've', 'với', 'xe'}
_EN_STOPWORDS = {'all', 'and', 'are', 'back', 'battery', 'beats', 'but', 'charging', 'cleaner', 'day', 'design', 'drop', 'for', 'feed', 'good', 'issue', 'is', 'last', 'launch', 'looks', 'matte', 'most', 'my', 'nice', 'price', 'quality', 'routine', 'service', 'slow', 'soft', 'still', 'support', 'the', 'than', 'this', 'tints', 'with'}
_ID_STOPWORDS = {'ada', 'aja', 'banget', 'bagus', 'dan', 'dengan', 'ini', 'itu', 'kerja', 'mau', 'nya', 'pabrik', 'produk', 'sangat', 'tidak', 'untuk', 'yang'}
_VI_DIACRITIC_MARKERS = 'ăâđêôơưàáạảãầấậẩẫằắặẳẵèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ'
_SOCIAL_SLANG_MAP = {'ko': 'không', 'k': 'không', 'hok': 'không', 'dc': 'được', 'đc': 'được', 'vs': 'với', 'sp': 'sản phẩm', 'nv': 'nhân viên'}
_LATIN_ENTITY_ALLOWLIST = {'vf', 'vinfast', 'byd', 'catl'}
_MEANINGFUL_EN_SHORTFORMS = {'flashsale', 'delivery', 'battery', 'charging', 'launch', 'support'}

@dataclass(frozen=True, slots=True)
class TextNormalizationResult:
    raw_text: str
    normalized_text: str
    normalized_text_compact: str
    urls: list[str]
    hashtags: list[str]
    emojis: list[str]
    language: str
    language_confidence: float
    language_provider: str
    language_provider_version: str
    language_model_id: str
    language_source: str
    language_metadata: dict[str, str | int | float | bool | None]
    language_supported: bool
    language_rejection_reason: str | None
    text_quality_label: str
    text_quality_flags: list[str]
    text_quality_score: float
    mixed_language_uncertain: bool
    semantic_route_hint: str

def normalize_whitespace(text):
    return WHITESPACE_RE.sub(' ', text).strip()

def extract_urls(text):
    return URL_RE.findall(text)

def extract_emojis(text):
    return [match.group(0) for match in EMOJI_RE.finditer(text)]

def normalize_hashtags(text, explicit_hashtags=None):
    hashtags = [match.lower() for match in HASHTAG_RE.findall(text)]
    if explicit_hashtags:
        hashtags.extend((tag.lower().lstrip('#') for tag in explicit_hashtags if tag))
    deduped = []
    for hashtag in hashtags:
        if hashtag not in deduped:
            deduped.append(hashtag)
    return deduped

def explicit_language_result(explicit_language):
    language = canonicalize_language_label(explicit_language)
    provenance = ProviderProvenance(provider_kind='language_id', provider_name='record_language_hint', provider_version='record-language-v2', model_id='record-explicit-language', device='cpu')
    return LanguageIdentificationResult(language=language, confidence=1.0 if language != 'unknown' else 0.0, provider_provenance=provenance, source=LanguageSource.EXPLICIT, metadata={'raw_explicit_language': explicit_language})

def resolve_language(text, *, explicit_language=None, language_id_provider=None):
    if explicit_language:
        return explicit_language_result(explicit_language)
    provider = language_id_provider or _HEURISTIC_LANGUAGE_PROVIDER
    return provider.detect(text)

def normalize_social_text(text, explicit_hashtags, explicit_language, *, language_id_provider=None):
    raw_text = normalize_whitespace(unicodedata.normalize('NFKC', text))
    urls = extract_urls(raw_text)
    hashtags = normalize_hashtags(raw_text, explicit_hashtags=explicit_hashtags)
    emojis = extract_emojis(raw_text)
    language_result = resolve_language(raw_text, explicit_language=explicit_language, language_id_provider=language_id_provider)
    normalized_text = _normalize_processing_text(raw_text, hashtags)
    normalized_text_compact = _compact_processing_text(normalized_text)
    language_label, language_supported, rejection_reason, gate_metadata = _language_gate(raw_text, normalized_text_compact, language_result)
    quality_flags = _text_quality_flags(raw_text=raw_text, normalized_text=normalized_text, compact_text=normalized_text_compact, urls=urls, emojis=emojis, language_supported=language_supported)
    text_quality_label = _text_quality_label(quality_flags)
    text_quality_score = _text_quality_score(quality_flags)
    return TextNormalizationResult(raw_text=raw_text, normalized_text=normalized_text, normalized_text_compact=normalized_text_compact, urls=urls, hashtags=hashtags, emojis=emojis, language=language_label, language_confidence=round(max(language_result.confidence, 0.72 if language_label in {'vi', 'en', 'mixed'} else 0.0), 4), language_provider=language_result.provider_provenance.provider_name, language_provider_version=language_result.provider_provenance.provider_version, language_model_id=language_result.provider_provenance.model_id, language_source=language_result.source.value, language_metadata={**{key: value for key, value in language_result.metadata.items() if isinstance(value, (str, int, float, bool)) or value is None}, **gate_metadata}, language_supported=language_supported, language_rejection_reason=rejection_reason, text_quality_label=text_quality_label, text_quality_flags=quality_flags, text_quality_score=text_quality_score, mixed_language_uncertain=language_label == 'mixed', semantic_route_hint='semantic_skip' if not language_supported else 'semantic_full')

def _normalize_processing_text(raw_text, hashtags):
    normalized = raw_text
    normalized = URL_RE.sub(' ', normalized)
    normalized = MENTION_RE.sub(' ', normalized)
    normalized = EMOJI_RE.sub(' ', normalized)
    for hashtag in hashtags:
        segmented = _segment_hashtag(hashtag)
        if segmented:
            normalized = normalized.replace(f'#{hashtag}', f' {segmented} ')
    normalized = REPEATED_ALPHA_RE.sub('\\1\\1', normalized)
    normalized = REPEATED_PUNCT_RE.sub('\\1', normalized)
    normalized = normalize_whitespace(normalized)
    for slang, replacement in _SOCIAL_SLANG_MAP.items():
        normalized = re.sub(f'(?<!\\w){re.escape(slang)}(?!\\w)', replacement, normalized, flags=re.IGNORECASE)
    return normalize_whitespace(normalized)

def _compact_processing_text(normalized_text):
    compact = PUNCT_COMPRESS_RE.sub(' ', normalized_text.casefold())
    compact = compact.replace('/', ' ')
    compact = compact.replace('_', ' ')
    return normalize_whitespace(compact)

def _segment_hashtag(hashtag):
    segmented = re.sub('([a-z])([A-Z])', '\\1 \\2', hashtag).replace('_', ' ').replace('-', ' ')
    segmented = re.sub('([A-Za-z])(\\d)', '\\1 \\2', segmented)
    segmented = re.sub('(\\d)([A-Za-z])', '\\1 \\2', segmented)
    return normalize_whitespace(segmented)

def _language_gate(raw_text, compact_text, provider_result):
    tokens = [token for token in TOKEN_RE.findall(compact_text) if token]
    if not tokens:
        return ('unsupported', False, 'no_semantic_content', {'gate_reason': 'no_semantic_content'})
    lower_tokens = [token.casefold() for token in tokens]
    token_count = max(len(lower_tokens), 1)
    vi_hits = sum((1 for token in lower_tokens if token in _VI_STOPWORDS))
    en_hits = sum((1 for token in lower_tokens if token in _EN_STOPWORDS))
    id_hits = sum((1 for token in lower_tokens if token in _ID_STOPWORDS))
    latin_entity_like = sum((1 for token in lower_tokens if _is_entity_like_token(token)))
    semantic_tokens = [token for token in lower_tokens if _is_semantic_token(token)]
    semantic_token_count = len(semantic_tokens)
    vi_semantic_hits = sum((1 for token in semantic_tokens if _supports_vi(token)))
    en_semantic_hits = sum((1 for token in semantic_tokens if token in _EN_STOPWORDS))
    ascii_semantic_hits = sum((1 for token in semantic_tokens if token.isascii() and len(token) >= 4 and (token not in _ID_STOPWORDS)))
    vi_score = vi_hits / token_count
    en_score = en_hits / token_count
    id_score = id_hits / token_count
    if any((character in raw_text.casefold() for character in _VI_DIACRITIC_MARKERS)):
        vi_score += 0.28
    provider_language = canonicalize_language_label(provider_result.language)
    if provider_language == 'vi':
        vi_score += 0.18
    elif provider_language == 'en':
        en_score += 0.16
    elif provider_language == 'mixed':
        vi_score += 0.08
        en_score += 0.08
    elif provider_language == 'id':
        id_score += 0.18
    unsupported_script = _unsupported_script_score(raw_text)
    unsupported_score = id_score + unsupported_script
    metadata = {'gate_provider_language': provider_language, 'vi_score': round(vi_score, 4), 'en_score': round(en_score, 4), 'unsupported_score': round(unsupported_score, 4), 'entity_like_token_count': latin_entity_like, 'semantic_token_count': semantic_token_count, 'vi_semantic_hits': vi_semantic_hits, 'en_semantic_hits': en_semantic_hits, 'ascii_semantic_hits': ascii_semantic_hits}
    if unsupported_score >= 0.42 and unsupported_score > vi_score + en_score + 0.08:
        return ('unsupported', False, 'unsupported_language', {**metadata, 'gate_reason': 'unsupported_language'})
    if semantic_token_count == 0:
        rejection = 'non_vi_en_latin_noise' if latin_entity_like else 'no_semantic_content'
        return ('unsupported', False, rejection, {**metadata, 'gate_reason': rejection})
    mixed_vi_evidence = vi_score >= 0.18 or vi_semantic_hits >= 2 or any((character in raw_text.casefold() for character in _VI_DIACRITIC_MARKERS))
    mixed_en_evidence = en_score >= 0.12 or en_semantic_hits >= 1
    if mixed_vi_evidence and mixed_en_evidence and (semantic_token_count >= 2) and (unsupported_score < 0.24):
        return ('mixed', True, None, {**metadata, 'gate_reason': 'accepted_mixed_verified'})
    if vi_semantic_hits >= 1 and ascii_semantic_hits >= 2 and (semantic_token_count >= 4) and (provider_language in {'en', 'mixed', 'unknown'}) and (unsupported_score < 0.18):
        return ('mixed', True, None, {**metadata, 'gate_reason': 'accepted_mixed_bridge'})
    if vi_score >= 0.2 or vi_semantic_hits >= 2:
        return ('vi', True, None, {**metadata, 'gate_reason': 'accepted_vi_verified'})
    if (en_score >= 0.2 and semantic_token_count >= 2 or (semantic_token_count == 1 and semantic_tokens[0] in _MEANINGFUL_EN_SHORTFORMS and (provider_language in {'en', 'mixed', 'unknown'}))) and unsupported_score < 0.18:
        return ('en', True, None, {**metadata, 'gate_reason': 'accepted_en_verified'})
    if latin_entity_like / token_count >= 0.5 and vi_score + en_score < 0.18:
        return ('unsupported', False, 'non_vi_en_latin_noise', {**metadata, 'gate_reason': 'non_vi_en_latin_noise'})
    if vi_score + en_score < 0.18 and semantic_token_count <= 2:
        return ('unsupported', False, 'low_vi_en_evidence', {**metadata, 'gate_reason': 'low_vi_en_evidence'})
    return ('unsupported', False, 'unsupported_language', {**metadata, 'gate_reason': 'unsupported_language'})

def _unsupported_script_score(text):
    unsupported_chars = 0
    total_letters = 0
    for char in text:
        if not char.isalpha():
            continue
        total_letters += 1
        codepoint = ord(char)
        if 2304 <= codepoint <= 2431 or 3584 <= codepoint <= 3711 or 2432 <= codepoint <= 2559 or (19968 <= codepoint <= 40959) or (12352 <= codepoint <= 12543) or (44032 <= codepoint <= 55215):
            unsupported_chars += 1
    if total_letters == 0:
        return 0.0
    return unsupported_chars / total_letters

def _is_entity_like_token(token):
    if any((character.isdigit() for character in token)):
        return True
    if token in _LATIN_ENTITY_ALLOWLIST:
        return True
    return len(token) <= 3 and token.isascii() and token.isalpha() and (token not in _EN_STOPWORDS) and (token not in _ID_STOPWORDS)

def _looks_supported_short_form(token):
    if not token or not token.isascii():
        return False
    if token in _ID_STOPWORDS:
        return False
    if any((character.isdigit() for character in token)):
        return True
    return len(token) >= 4 and token.isalpha()

def _is_semantic_token(token):
    if token in _LATIN_ENTITY_ALLOWLIST:
        return False
    if any((character.isdigit() for character in token)):
        return False
    if len(token) < 2:
        return False
    return token.isalpha()

def _supports_vi(token):
    if token in _VI_STOPWORDS:
        return True
    return any((character in _VI_DIACRITIC_MARKERS for character in token))

def _text_quality_flags(*, raw_text, normalized_text, compact_text, urls, emojis, language_supported):
    tokens = [token for token in compact_text.split() if token not in {'url', 'mention'}]
    flags = []
    if not language_supported:
        flags.append('unsupported_language')
    if len(tokens) <= 2 and len(emojis) >= 2:
        flags.append('emoji_dominant')
    if len(tokens) <= 2 and (not urls):
        flags.append('low_information')
    if not tokens and raw_text:
        flags.append('reaction_only')
    if len(urls) >= 2 or (urls and len(tokens) <= 3):
        flags.append('url_heavy')
    if len(set(tokens)) <= 2 and len(tokens) >= 4:
        flags.append('template_like')
    if len(tokens) <= 3 and normalized_text.casefold() in {'ok', 'oke', 'haha', 'hehe', 'nice'}:
        flags.append('reaction_only')
    if not tokens:
        flags.append('no_semantic_content')
    return sorted(set(flags))

def _text_quality_label(flags):
    if 'unsupported_language' in flags:
        return 'unsupported_language'
    if 'no_semantic_content' in flags:
        return 'no_semantic_content'
    if 'reaction_only' in flags:
        return 'reaction_only'
    if 'emoji_dominant' in flags or 'low_information' in flags:
        return 'low_information'
    if 'url_heavy' in flags or 'template_like' in flags:
        return 'spam_like'
    return 'normal'

def _text_quality_score(flags):
    score = 1.0
    penalties = {'unsupported_language': 0.9, 'reaction_only': 0.45, 'low_information': 0.3, 'url_heavy': 0.2, 'template_like': 0.2, 'emoji_dominant': 0.15}
    for flag in flags:
        score -= penalties.get(flag, 0.0)
    return round(max(score, 0.05), 4)
