from __future__ import annotations
from smap.bi.models import BIReportBundle, InsightCardBundle
from smap.bi.reports import build_bi_reports
from smap.insights.models import InsightCard
from smap.marts.builder import MartBundle

def generate_insight_bundle(reports):
    return reports.insight_card_bundle

def generate_insights(source):
    reports = source if isinstance(source, BIReportBundle) else build_bi_reports(source)
    return list(generate_insight_bundle(reports).cards)
