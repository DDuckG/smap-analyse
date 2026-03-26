from __future__ import annotations
import re
from smap.enrichers.semantic_models import SemanticSegment
_HARD_BOUNDARY_RE = re.compile('[,.!?;\\n]+|[|*]+')
_CONTRAST_BOUNDARY_RE = re.compile('\\b(?:nhung|nhưng|but|however|though|although|tuy nhien|tuy nhiên|con|còn|ma|mà)\\b', flags=re.IGNORECASE | re.UNICODE)

def normalize_for_scope(text):
    return ' '.join(text.lower().split())

class SemanticSegmenter:

    def segment(self, text):
        if not text.strip():
            return []
        boundaries = sorted([*(match.span() + ('hard',) for match in _HARD_BOUNDARY_RE.finditer(text)), *(match.span() + ('contrast',) for match in _CONTRAST_BOUNDARY_RE.finditer(text))], key=lambda item: item[0])
        segments: list[SemanticSegment] = []
        cursor = 0
        segment_index = 0
        pending_boundary: str | None = None
        for start, end, boundary_type in boundaries:
            if start > cursor:
                segment_text = text[cursor:start].strip(' ,')
                if segment_text:
                    segment_start = text.index(segment_text, cursor, start)
                    boundary_text = text[start:end]
                    segments.append(SemanticSegment(segment_id=f'seg-{segment_index}', start=segment_start, end=segment_start + len(segment_text), text=segment_text, normalized_text=normalize_for_scope(segment_text), leading_boundary=pending_boundary, contrastive=pending_boundary == 'contrast', question_like='?' in segment_text or '?' in boundary_text))
                    segment_index += 1
            cursor = end
            pending_boundary = boundary_type
        trailing = text[cursor:].strip(' ,')
        if trailing:
            trailing_start = text.rfind(trailing, cursor)
            segments.append(SemanticSegment(segment_id=f'seg-{segment_index}', start=trailing_start, end=trailing_start + len(trailing), text=trailing, normalized_text=normalize_for_scope(trailing), leading_boundary=pending_boundary, contrastive=pending_boundary == 'contrast', question_like='?' in trailing or text.strip().endswith('?')))
        if not segments:
            stripped = text.strip()
            return [SemanticSegment(segment_id='seg-0', start=text.index(stripped), end=text.index(stripped) + len(stripped), text=stripped, normalized_text=normalize_for_scope(stripped), question_like='?' in stripped)]
        return segments
