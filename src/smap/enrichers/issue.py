from __future__ import annotations

from smap.enrichers.models import FactProvenance, IssueSeverity, IssueSignalFact
from smap.enrichers.semantic_models import EvidenceMode
from smap.normalization.models import MentionRecord
from smap.threads.models import MentionContext

ISSUE_MARKERS = {
    "defect": {"lỗi", "hỏng", "defect"},
    "service_issue": {"dịch vụ", "service", "bảo hành"},
    "safety_issue": {"nguy hiểm", "an toàn", "safety"},
    "supply_issue": {"chờ", "trễ", "delivery"},
    "trust_issue": {"lùa", "lừa", "trust"},
}


class IssueSignalEnricher:
    name = "issue_signal"

    def enrich(self, mention: MentionRecord, context: MentionContext | None) -> list[IssueSignalFact]:
        text = mention.normalized_text
        facts: list[IssueSignalFact] = []
        for issue_category, markers in ISSUE_MARKERS.items():
            if any(marker in text for marker in markers):
                severity: IssueSeverity
                if mention.depth == 0:
                    severity = "high"
                elif mention.likes is not None and mention.likes > 20:
                    severity = "medium"
                else:
                    severity = "low"
                facts.append(
                    IssueSignalFact(
                        mention_id=mention.mention_id,
                        source_uap_id=mention.source_uap_id,
                        issue_category=issue_category,
                        severity=severity,
                        confidence=0.66,
                        evidence_mode=EvidenceMode.DIRECT_OBSERVATION,
                        provenance=FactProvenance(
                            source_uap_id=mention.source_uap_id,
                            mention_id=mention.mention_id,
                            provider_version="issue-rules-v1",
                            rule_version="issue-rules-v1",
                            evidence_text=context.context_text if context else mention.raw_text,
                        ),
                    )
                )
        return facts
