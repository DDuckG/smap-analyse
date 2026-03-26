from __future__ import annotations
import re
from collections.abc import Iterable
from smap.canonicalization.alias import boundary_contains, boundary_spans, normalize_alias
from smap.enrichers.semantic_models import AnchorType, SemanticAnchor, SemanticSegment
SeedSet = dict[str, tuple[str, ...]]
ASPECT_SEEDS = {'price': ('price', 'giá', 'gia', 'đắt', 'dat', 'rẻ', 're', 'value', 'cost'), 'quality': ('quality', 'chất lượng', 'chat luong', 'build', 'finish', 'texture'), 'design': ('design', 'thiết kế', 'thiet ke', 'look', 'style', 'đẹp', 'dep', 'mã'), 'performance': ('performance', 'hiệu năng', 'hieu nang', 'lag', 'slow', 'nhanh', 'chậm', 'cham'), 'durability': ('durable', 'durability', 'bền', 'ben', 'wear', 'broken'), 'availability': ('stock', 'available', 'availability', 'hết hàng', 'het hang', 'sold out'), 'customer_service': ('service', 'support', 'cskh', 'bảo hành', 'bao hanh', 'refund'), 'delivery': ('delivery', 'ship', 'shipping', 'giao', 'arrive', 'delay', 'trễ', 'tre'), 'trust': ('trust', 'fake', 'uy tín', 'uy tin', 'scam', 'legit'), 'safety': ('safety', 'safe', 'an toàn', 'an toan', 'unsafe', 'danger'), 'battery': ('battery', 'pin'), 'charging': ('charging', 'charge', 'sạc', 'sac'), 'usability': ('easy', 'hard to use', 'khó dùng', 'kho dung', 'confusing', 'usability')}
ISSUE_SEEDS = {'product_defect': ('defect', 'broken', 'lỗi', 'loi', 'hỏng', 'hong', 'fail'), 'performance_problem': ('lag', 'slow', 'chậm', 'cham', 'freeze', 'crash', 'nóng', 'nong'), 'service_problem': ('service', 'support', 'bảo hành', 'bao hanh', 'refund'), 'delivery_problem': ('delivery', 'ship', 'delay', 'trễ', 'tre', 'giao'), 'trust_concern': ('fake', 'scam', 'lừa', 'lua', 'suspicious', 'trust'), 'safety_concern': ('unsafe', 'danger', 'nguy hiểm', 'nguy hiem', 'cháy', 'chay'), 'compliance_concern': ('policy', 'compliance', 'ban', 'legal', 'regulation'), 'pricing_value_concern': ('overpriced', 'expensive', 'đắt', 'dat', 'pricey', 'value'), 'usability_problem': ('hard to use', 'confusing', 'khó dùng', 'kho dung', 'buggy flow'), 'availability_problem': ('sold out', 'hết hàng', 'het hang', 'unavailable', 'backorder')}
POSITIVE_CUES = {'good': 0.7, 'great': 0.9, 'love': 1.0, 'nice': 0.7, 'smooth': 0.75, 'đẹp': 0.75, 'dep': 0.75, 'ổn': 0.65, 'on': 0.65, 'ok': 0.55, 'xịn': 0.85, 'xin': 0.85, 'nhanh': 0.7}
NEGATIVE_CUES = {'bad': -0.7, 'hate': -1.0, 'slow': -0.8, 'lag': -0.85, 'delay': -0.75, 'confusing': -0.65, 'chậm': -0.8, 'cham': -0.8, 'kém': -0.75, 'kem': -0.75, 'tệ': -0.85, 'te': -0.85, 'lỗi': -0.85, 'loi': -0.85, 'khó': -0.65, 'kho': -0.65, 'đắt': -0.65, 'dat': -0.65}
NEGATION_CUES = ('không', 'khong', 'not', 'never', 'chưa', 'chua', 'chẳng', 'chang')
UNCERTAINTY_CUES = ('?', 'có ai', 'co ai', 'hình như', 'hinh nhu', 'maybe', 'perhaps', 'seems', 'có vẻ', 'co ve')
COMPARISON_CUES = ('hơn', 'hon', 'better', 'worse', 'than', 'compared')
HEARSAY_CUES = ('nghe nói', 'nghe noi', 'đồn', 'don', 'apparently', 'people say', 'rumor')
ESCALATION_CUES = ('report', 'refund', 'never again', 'quá tệ', 'qua te', 'gấp', 'urgent', 'danger')
FIRST_PERSON_CUES = ('mình', 'minh', 'tôi', 'toi', 'em', 'i', 'my', 'me')
_WORD_BOUNDARY = re.compile('(?<!\\w)(?P<token>[\\wÀ-ỹ]+(?:\\s+[\\wÀ-ỹ]+){0,2})(?!\\w)', flags=re.UNICODE)

def extract_lexical_anchors(segment, *, aspect_seeds=None, issue_seeds=None):
    anchors = []
    anchors.extend(_anchors_from_seeds(segment, aspect_seeds or ASPECT_SEEDS, AnchorType.ASPECT, 0.7))
    anchors.extend(_anchors_from_seeds(segment, issue_seeds or ISSUE_SEEDS, AnchorType.ISSUE, 0.76))
    anchors.extend(_anchors_from_scores(segment, POSITIVE_CUES, positive=True))
    anchors.extend(_anchors_from_scores(segment, NEGATIVE_CUES, positive=False))
    anchors.extend(_cue_anchors(segment, NEGATION_CUES, AnchorType.NEGATION, 0.72))
    anchors.extend(_cue_anchors(segment, UNCERTAINTY_CUES, AnchorType.UNCERTAINTY, 0.68))
    anchors.extend(_cue_anchors(segment, COMPARISON_CUES, AnchorType.COMPARISON, 0.65))
    anchors.extend(_cue_anchors(segment, HEARSAY_CUES, AnchorType.HEARSAY, 0.74))
    anchors.extend(_cue_anchors(segment, ESCALATION_CUES, AnchorType.ESCALATION, 0.8))
    if segment.contrastive:
        anchors.append(SemanticAnchor(anchor_type=AnchorType.CONTRAST, label='contrast_boundary', normalized_text=segment.leading_boundary or 'contrast', start=segment.start, end=segment.start, text=segment.leading_boundary or '', source='segmenter', confidence=0.8, segment_id=segment.segment_id))
    return sorted(anchors, key=lambda item: (item.start, item.end, item.anchor_type.value))

def build_target_anchors(*, mention_text, segment, target_text, canonical_entity_id, concept_entity_id, unresolved_cluster_id, entity_type, source, inherited=False, inherited_from_mention_id=None):
    anchors = []
    for start, end in boundary_spans(mention_text, target_text):
        if start < segment.start or end > segment.end:
            continue
        anchors.append(SemanticAnchor(anchor_type=AnchorType.TARGET, label=canonical_entity_id or concept_entity_id or unresolved_cluster_id or normalize_alias(target_text) or target_text.casefold(), normalized_text=normalize_alias(target_text) or target_text.casefold(), start=start, end=end, text=mention_text[start:end], source=source, confidence=0.8 if not inherited else 0.55, segment_id=segment.segment_id, metadata={'canonical_entity_id': canonical_entity_id or '', 'concept_entity_id': concept_entity_id or '', 'unresolved_cluster_id': unresolved_cluster_id or '', 'entity_type': entity_type or '', 'target_text': target_text, 'inherited': inherited, 'inherited_from_mention_id': inherited_from_mention_id or ''}))
    return anchors

def segment_tokens(segment):
    tokens = []
    for match in _WORD_BOUNDARY.finditer(segment.text):
        token = match.group('token')
        start = segment.start + match.start('token')
        end = segment.start + match.end('token')
        tokens.append((normalize_alias(token), start, end))
    return tokens

def contains_first_person(segment):
    return any((boundary_contains(segment.text, cue) for cue in FIRST_PERSON_CUES))

def _anchors_from_seeds(segment, seeds, anchor_type, confidence):
    anchors = []
    for label, variants in seeds.items():
        for variant in variants:
            anchors.extend(_cue_anchors(segment, (variant,), anchor_type, confidence, label=label))
    return anchors

def _anchors_from_scores(segment, cues, *, positive):
    anchors = []
    for cue, score in cues.items():
        for anchor in _cue_anchors(segment, (cue,), AnchorType.POLARITY, min(0.92, 0.55 + abs(score) * 0.25), label='positive' if positive else 'negative'):
            anchor.polarity = score
            anchors.append(anchor)
    return anchors

def _cue_anchors(segment, cues, anchor_type, confidence, *, label=None):
    anchors = []
    for cue in cues:
        for start, end in boundary_spans(segment.text, cue):
            absolute_start = segment.start + start
            absolute_end = segment.start + end
            anchors.append(SemanticAnchor(anchor_type=anchor_type, label=label or cue, normalized_text=normalize_alias(cue) or cue.casefold(), start=absolute_start, end=absolute_end, text=segment.text[start:end], source='lexical_seed', confidence=confidence, segment_id=segment.segment_id))
    return anchors
