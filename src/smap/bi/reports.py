from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import cast
import polars as pl
from smap.analytics.metrics import build_metrics
from smap.bi.contracts import ReportContract, bi_contract_registry
from smap.bi.models import BIReportBundle, BuzzEntityRow, BuzzReport, BuzzTopicRow, CreatorBreakdownRow, CreatorSourceBreakdownReport, EmergingTopicRow, EmergingTopicsReport, InsightCardBundle, ReportWindow, SourceBreakdownRow, SOVReport, SOVRow, ThreadControversyReport, ThreadControversyRow, ThreadInitiatorRow, TopIssueRow, TopIssuesReport, UnresolvedSummary
from smap.insights.models import InsightCard
from smap.marts.builder import MartBundle

@dataclass(frozen=True, slots=True)
class _WindowContext:
    window: ReportWindow
    current_ids: set[str]
    previous_ids: set[str]

def build_bi_reports(bundle, *, metrics=None, weighting_mode=None):
    metrics = metrics or build_metrics(bundle)
    diagnostics = cast(dict[str, object], metrics.get('diagnostics', {}))
    weighting_mode = weighting_mode or str(diagnostics.get('active_weighting_mode', 'raw'))
    contracts = bi_contract_registry()
    window_context = _window_context(bundle.tables['fact_mentions'])
    sov_report = _build_sov_report(bundle, contracts['sov_report'], window_context, weighting_mode=weighting_mode)
    buzz_report = _build_buzz_report(bundle, contracts['buzz_report'], window_context, weighting_mode=weighting_mode)
    emerging_topics_report = _build_emerging_topics_report(bundle, contracts['emerging_topics_report'], window_context.window)
    top_issues_report = _build_top_issues_report(bundle, contracts['top_issues_report'], window_context)
    thread_controversy_report = _build_thread_controversy_report(bundle, contracts['thread_controversy_report'], window_context.window)
    creator_source_breakdown_report = _build_creator_source_breakdown_report(bundle, contracts['creator_source_breakdown_report'], window_context.window, thread_controversy_report=thread_controversy_report)
    insight_card_bundle = _build_insight_card_bundle(contracts['insight_card_bundle'], window_context.window, sov_report=sov_report, buzz_report=buzz_report, emerging_topics_report=emerging_topics_report, top_issues_report=top_issues_report, thread_controversy_report=thread_controversy_report, creator_source_breakdown_report=creator_source_breakdown_report)
    return BIReportBundle(sov_report=sov_report, buzz_report=buzz_report, emerging_topics_report=emerging_topics_report, top_issues_report=top_issues_report, thread_controversy_report=thread_controversy_report, creator_source_breakdown_report=creator_source_breakdown_report, insight_card_bundle=insight_card_bundle)

def _window_context(fact_mentions):
    if fact_mentions.is_empty() or 'posted_at' not in fact_mentions.columns:
        return _WindowContext(window=ReportWindow(delta_kind='none', delta_comparison_mode='none'), current_ids=set(), previous_ids=set())
    valid = fact_mentions.filter(pl.col('posted_at').is_not_null()).sort('posted_at')
    if valid.is_empty():
        return _WindowContext(window=ReportWindow(delta_kind='none', delta_comparison_mode='none'), current_ids=set(), previous_ids=set())
    analysis_start = _iso(valid['posted_at'].to_list()[0])
    analysis_end = _iso(valid['posted_at'].to_list()[-1])
    if valid.height < 2:
        return _WindowContext(window=ReportWindow(analysis_start=analysis_start, analysis_end=analysis_end, delta_kind='none', delta_comparison_mode='none'), current_ids=set(), previous_ids=set())
    split_index = max(valid.height // 2, 1)
    previous = valid.slice(0, split_index)
    current = valid.slice(split_index, valid.height - split_index)
    if current.is_empty():
        return _WindowContext(window=ReportWindow(analysis_start=analysis_start, analysis_end=analysis_end, delta_kind='none', delta_comparison_mode='none'), current_ids=set(), previous_ids=set())
    return _WindowContext(window=ReportWindow(analysis_start=analysis_start, analysis_end=analysis_end, delta_current_start=_iso(current['posted_at'].to_list()[0]), delta_current_end=_iso(current['posted_at'].to_list()[-1]), delta_previous_start=_iso(previous['posted_at'].to_list()[0]), delta_previous_end=_iso(previous['posted_at'].to_list()[-1]), delta_kind='intra_window_delta', delta_comparison_mode='chronological_half_split'), current_ids=set(current['mention_id'].to_list()), previous_ids=set(previous['mention_id'].to_list()))

def _build_sov_report(bundle, contract, window_context, *, weighting_mode):
    fact_mentions = bundle.tables['fact_mentions']
    fact_entity_mentions = bundle.tables['fact_entity_mentions']
    dim_entities = bundle.tables['dim_entities']
    mention_volume = int(fact_mentions.height)
    effective_mention_volume = _effective_mention_volume(fact_mentions, weighting_mode)
    mention_weights = _mention_weights(fact_mentions, weighting_mode)
    resolved = fact_entity_mentions.filter(pl.col('canonical_entity_id').is_not_null()) if not fact_entity_mentions.is_empty() else pl.DataFrame()
    entity_dims = {row['canonical_entity_id']: row for row in dim_entities.to_dicts()} if not dim_entities.is_empty() and 'canonical_entity_id' in dim_entities.columns else {}
    current_rows = _entity_counts(resolved, mention_ids=window_context.current_ids)
    previous_rows = _entity_counts(resolved, mention_ids=window_context.previous_ids)
    current_effective = _weighted_entity_counts(resolved, mention_weights, mention_ids=window_context.current_ids)
    previous_effective = _weighted_entity_counts(resolved, mention_weights, mention_ids=window_context.previous_ids)
    current_total_hits = max(sum((row['entity_hit_count'] for row in current_rows.values())), 1)
    previous_total_hits = max(sum((row['entity_hit_count'] for row in previous_rows.values())), 1)
    effective_rows = _weighted_entity_counts(resolved, mention_weights)
    rows = []
    if not resolved.is_empty():
        full_grouped = resolved.group_by('canonical_entity_id').agg(pl.col('mention_id').n_unique().alias('mention_count'), pl.len().alias('entity_hit_count')).sort(['mention_count', 'entity_hit_count', 'canonical_entity_id'], descending=[True, True, False])
        resolved_mention_total = int(resolved.select(pl.col('mention_id').n_unique()).item())
        for row in full_grouped.to_dicts():
            entity_id = row['canonical_entity_id']
            current_row = current_rows.get(entity_id, {'mention_count': 0, 'entity_hit_count': 0})
            previous_row = previous_rows.get(entity_id, {'mention_count': 0, 'entity_hit_count': 0})
            effective_count = effective_rows.get(entity_id, 0.0)
            current_effective_count = current_effective.get(entity_id, 0.0)
            previous_effective_count = previous_effective.get(entity_id, 0.0)
            mention_share = _ratio(int(row['mention_count']), mention_volume)
            entity_hit_share = _ratio(int(row['entity_hit_count']), int(resolved.height))
            resolved_mention_share = _ratio(int(row['mention_count']), resolved_mention_total)
            rows.append(SOVRow(canonical_entity_id=entity_id, entity_name=_entity_name(entity_dims, entity_id), entity_type=_entity_type(entity_dims, entity_id), mention_count=int(row['mention_count']), mention_share=mention_share, effective_mention_count=round(effective_count, 4), effective_mention_share=_ratio(effective_count, effective_mention_volume), entity_hit_count=int(row['entity_hit_count']), entity_hit_share=entity_hit_share, resolved_mention_count=int(row['mention_count']), resolved_mention_share=resolved_mention_share, delta_mention_count=int(current_row['mention_count'] - previous_row['mention_count']), delta_mention_share=round(_ratio(current_effective_count, max(sum(current_effective.values()), 1.0)) - _ratio(previous_effective_count, max(sum(previous_effective.values()), 1.0)), 4), delta_entity_hit_count=int(current_row['entity_hit_count'] - previous_row['entity_hit_count']), delta_entity_hit_share=round(_ratio(current_row['entity_hit_count'], current_total_hits) - _ratio(previous_row['entity_hit_count'], previous_total_hits), 4), evidence_references=_head_mentions(resolved.filter(pl.col('canonical_entity_id') == entity_id), limit=5)))
    rows.sort(key=lambda item: (-item.effective_mention_share if weighting_mode != 'raw' else -item.mention_share, -item.entity_hit_share, item.canonical_entity_id))
    unresolved_mentions = mention_volume - (int(resolved.select(pl.col('mention_id').n_unique()).item()) if not resolved.is_empty() else 0)
    unresolved_reason_breakdown = _entity_unresolved_reason_breakdown(fact_entity_mentions)
    return SOVReport(contract=contract, window=window_context.window, weighting_mode=weighting_mode, effective_mention_volume=effective_mention_volume, filters_used={'weighting_mode': weighting_mode}, unresolved_summary=UnresolvedSummary(count=max(unresolved_mentions, 0), ratio=_ratio(max(unresolved_mentions, 0), mention_volume), notes='unresolved mentions remain in the slice denominator and are excluded from entity rows', reason_breakdown=unresolved_reason_breakdown), entities=rows[:10])

def _build_buzz_report(bundle, contract, window_context, *, weighting_mode):
    fact_mentions = bundle.tables['fact_mentions']
    fact_entity_mentions = bundle.tables['fact_entity_mentions']
    fact_topics = bundle.tables['fact_topics']
    fact_topic_artifacts = bundle.tables['fact_topic_artifacts']
    dim_entities = bundle.tables['dim_entities']
    mention_volume = max(int(fact_mentions.height), 1)
    effective_mention_volume = max(_effective_mention_volume(fact_mentions, weighting_mode), 1.0)
    mention_weights = _mention_weights(fact_mentions, weighting_mode)
    entity_dims = {row['canonical_entity_id']: row for row in dim_entities.to_dicts()} if not dim_entities.is_empty() and 'canonical_entity_id' in dim_entities.columns else {}
    entity_rows = []
    resolved = fact_entity_mentions.filter(pl.col('canonical_entity_id').is_not_null()) if not fact_entity_mentions.is_empty() else pl.DataFrame()
    current_counts = _entity_counts(resolved, mention_ids=window_context.current_ids)
    previous_counts = _entity_counts(resolved, mention_ids=window_context.previous_ids)
    effective_entity_counts = _weighted_entity_counts(resolved, mention_weights)
    current_effective_entity_counts = _weighted_entity_counts(resolved, mention_weights, mention_ids=window_context.current_ids)
    previous_effective_entity_counts = _weighted_entity_counts(resolved, mention_weights, mention_ids=window_context.previous_ids)
    if not resolved.is_empty():
        grouped = resolved.group_by('canonical_entity_id').agg(pl.col('mention_id').n_unique().alias('mention_count')).sort(['mention_count', 'canonical_entity_id'], descending=[True, False])
        for row in grouped.to_dicts():
            entity_id = row['canonical_entity_id']
            current_count = int(current_counts.get(entity_id, {'mention_count': 0})['mention_count'])
            previous_count = int(previous_counts.get(entity_id, {'mention_count': 0})['mention_count'])
            delta_count = current_count - previous_count
            mention_count = int(row['mention_count'])
            effective_count = effective_entity_counts.get(entity_id, 0.0)
            current_effective_count = current_effective_entity_counts.get(entity_id, 0.0)
            previous_effective_count = previous_effective_entity_counts.get(entity_id, 0.0)
            growth_ratio = _growth_ratio(current_effective_count, previous_effective_count)
            entity_rows.append(BuzzEntityRow(canonical_entity_id=entity_id, entity_name=_entity_name(entity_dims, entity_id), entity_type=_entity_type(entity_dims, entity_id), mention_count=mention_count, mention_share=_ratio(mention_count, mention_volume), effective_mention_count=round(effective_count, 4), effective_mention_share=_ratio(effective_count, effective_mention_volume), delta_count=delta_count, growth_ratio_proxy=growth_ratio, buzz_score_proxy=round(effective_count * (1.0 + max(growth_ratio, 0.0)), 4), evidence_references=_head_mentions(resolved.filter(pl.col('canonical_entity_id') == entity_id), limit=5)))
        entity_rows.sort(key=lambda item: (-item.buzz_score_proxy, -item.effective_mention_count, item.canonical_entity_id))
    artifact_by_reporting_key = _artifact_map(fact_topic_artifacts)
    topic_rows = []
    suppressed = 0
    if not fact_topics.is_empty():
        reporting_topics = _topic_reporting_frame(fact_topics)
        effective_topic_counts = _weighted_topic_counts(reporting_topics, mention_weights)
        grouped = reporting_topics.group_by(['reporting_topic_key', 'reporting_topic_label']).agg(pl.col('mention_id').n_unique().alias('mention_count'), pl.col('topic_key').unique().alias('raw_topic_keys')).sort(['mention_count', 'reporting_topic_key'], descending=[True, False])
        current_topic_counts = _topic_counts(reporting_topics, mention_ids=window_context.current_ids)
        previous_topic_counts = _topic_counts(reporting_topics, mention_ids=window_context.previous_ids)
        current_effective_topic_counts = _weighted_topic_counts(reporting_topics, mention_weights, mention_ids=window_context.current_ids)
        previous_effective_topic_counts = _weighted_topic_counts(reporting_topics, mention_weights, mention_ids=window_context.previous_ids)
        stable_candidates = []
        all_candidates = []
        for row in grouped.to_dicts():
            reporting_key = row['reporting_topic_key']
            artifact = artifact_by_reporting_key.get(reporting_key)
            current_count = current_topic_counts.get(reporting_key, 0)
            previous_count = previous_topic_counts.get(reporting_key, 0)
            delta_count = current_count - previous_count
            effective_count = effective_topic_counts.get(reporting_key, 0.0)
            growth_ratio = _growth_ratio(current_effective_topic_counts.get(reporting_key, 0.0), previous_effective_topic_counts.get(reporting_key, 0.0))
            topic_row = BuzzTopicRow(reporting_topic_key=reporting_key, reporting_topic_label=row['reporting_topic_label'], label_source=str(artifact.get('label_source')) if artifact and artifact.get('label_source') is not None else None, label_health_score=_optional_float(artifact.get('label_health_score')) if artifact else None, reporting_status=str(artifact.get('reporting_status') or 'reportable') if artifact else 'reportable', raw_topic_keys=[value for value in cast(list[object], row['raw_topic_keys']) if isinstance(value, str)], canonical_evidence_phrases=[value for value in cast(list[object], artifact.get('canonical_evidence_phrases') or []) if isinstance(value, str)] if artifact else [], salient_terms=[value for value in cast(list[object], artifact.get('salient_terms') or []) if isinstance(value, str)] if artifact else [], supporting_phrases=[value for value in cast(list[object], artifact.get('supporting_phrases') or []) if isinstance(value, str)] if artifact else [], mention_count=int(row['mention_count']), mention_share=_ratio(int(row['mention_count']), mention_volume), effective_mention_count=round(effective_count, 4), effective_mention_share=_ratio(effective_count, effective_mention_volume), delta_count=delta_count, growth_ratio_proxy=growth_ratio, buzz_score_proxy=round(effective_count * (1.0 + max(growth_ratio, 0.0)), 4), quality_score=_optional_float(artifact.get('quality_score')) if artifact else None, usefulness_score=_optional_float(artifact.get('usefulness_score')) if artifact else None, reportability_score=_optional_float(artifact.get('reportability_score')) if artifact else None, artifact_purity_score=_optional_float(artifact.get('artifact_purity_score')) if artifact else None, weak_topic=bool(artifact.get('weak_topic')) if artifact else False, noisy_topic=bool(artifact.get('noisy_topic')) if artifact else False, evidence_references=_head_mentions(reporting_topics.filter(pl.col('reporting_topic_key') == reporting_key), limit=5))
            all_candidates.append(topic_row)
            if topic_row.reporting_status != 'reportable' or topic_row.weak_topic or topic_row.noisy_topic:
                suppressed += 1
            else:
                stable_candidates.append(topic_row)
        topic_rows = stable_candidates if stable_candidates else all_candidates
        topic_rows.sort(key=lambda item: (-item.buzz_score_proxy, -item.effective_mention_count, item.reporting_topic_key))
    return BuzzReport(contract=contract, window=window_context.window, weighting_mode=weighting_mode, effective_mention_volume=effective_mention_volume, filters_used={'weighting_mode': weighting_mode}, suppressed_weak_noisy_topics=suppressed, entity_buzz=entity_rows[:10], topic_buzz=topic_rows[:10])

def _build_emerging_topics_report(bundle, contract, window):
    fact_topic_artifacts = bundle.tables['fact_topic_artifacts']
    topics = []
    if not fact_topic_artifacts.is_empty():
        emerging = fact_topic_artifacts.filter(pl.col('emerging_topic') & (pl.col('reporting_status') == 'reportable')).sort(['growth_delta', 'quality_score', 'reporting_topic_key'], descending=[True, True, False])
        for row in emerging.to_dicts():
            topics.append(EmergingTopicRow(reporting_topic_key=row.get('reporting_topic_key') or row['topic_key'], reporting_topic_label=row.get('reporting_topic_label') or row['topic_label'], label_source=str(row.get('label_source')) if row.get('label_source') is not None else None, label_health_score=_optional_float(row.get('label_health_score')), effective_topic_key=row.get('effective_topic_key'), effective_topic_label=row.get('effective_topic_label'), raw_topic_key=row['topic_key'], raw_topic_label=row['topic_label'], topic_size=int(row['topic_size']), growth_delta=float(row['growth_delta']) if row.get('growth_delta') is not None else None, stability_score=float(row['stability_score']) if row.get('stability_score') is not None else None, quality_score=float(row['quality_score']) if row.get('quality_score') is not None else None, usefulness_score=float(row['usefulness_score']) if row.get('usefulness_score') is not None else None, reportability_score=float(row['reportability_score']) if row.get('reportability_score') is not None else None, artifact_purity_score=float(row['artifact_purity_score']) if row.get('artifact_purity_score') is not None else None, reporting_status=str(row.get('reporting_status') or 'reportable'), weak_topic=bool(row.get('weak_topic')), noisy_topic=bool(row.get('noisy_topic')), quality_reason_flags=list(row.get('quality_reason_flags') or []), top_terms=list(row.get('top_terms') or []), canonical_evidence_phrases=list(row.get('canonical_evidence_phrases') or []), salient_terms=list(row.get('salient_terms') or []), supporting_phrases=list(row.get('supporting_phrases') or []), representative_document_ids=list(row.get('representative_document_ids') or []), representative_texts=list(row.get('representative_texts') or [])))
    return EmergingTopicsReport(contract=contract, window=window, topics=topics[:10])

def _build_top_issues_report(bundle, contract, window_context):
    fact_issue_signals = bundle.tables['fact_issue_signals']
    mention_total = max(int(bundle.tables['fact_mentions'].height), 1)
    issues = []
    if not fact_issue_signals.is_empty():
        valid = fact_issue_signals.filter(pl.col('issue_category').is_not_null())
        current = valid.filter(pl.col('mention_id').is_in(list(window_context.current_ids))) if window_context.current_ids else valid.head(0)
        previous = valid.filter(pl.col('mention_id').is_in(list(window_context.previous_ids))) if window_context.previous_ids else valid.head(0)
        current_counts = _issue_mention_counts(current)
        previous_counts = _issue_mention_counts(previous)
        for issue_category, issue_frame in valid.partition_by('issue_category', as_dict=True).items():
            issue_label = issue_category[0] if isinstance(issue_category, tuple) else issue_category
            issue_fact_count = int(issue_frame.height)
            mention_count = int(issue_frame.select(pl.col('mention_id').n_unique()).item())
            severity_mix = _mix(issue_frame, 'severity')
            evidence_mode_mix = _mix(issue_frame, 'evidence_mode')
            pressure = _issue_pressure_proxy(issue_frame)
            issues.append(TopIssueRow(issue_category=str(issue_label), issue_fact_count=issue_fact_count, mention_count=mention_count, mention_prevalence_ratio=_ratio(mention_count, mention_total), issue_pressure_proxy=pressure, growth_delta_mentions=current_counts.get(str(issue_label), 0) - previous_counts.get(str(issue_label), 0), severity_mix=severity_mix, evidence_mode_mix=evidence_mode_mix, evidence_references=_head_mentions(issue_frame, limit=5)))
        issues.sort(key=lambda item: (-item.issue_pressure_proxy, -item.mention_count, item.issue_category))
    return TopIssuesReport(contract=contract, window=window_context.window, issues=issues[:10])

def _build_thread_controversy_report(bundle, contract, window):
    fact_threads = bundle.tables['fact_threads']
    fact_mentions = bundle.tables['fact_mentions']
    fact_sentiment = bundle.tables['fact_sentiment']
    fact_issue_signals = bundle.tables['fact_issue_signals']
    threads = []
    if not fact_threads.is_empty():
        sentiments = fact_sentiment.select(['mention_id', 'sentiment']) if not fact_sentiment.is_empty() and 'mention_id' in fact_sentiment.columns else pl.DataFrame({'mention_id': [], 'sentiment': []})
        mention_roots = fact_mentions.select(['mention_id', 'root_id', 'author_id']) if not fact_mentions.is_empty() else pl.DataFrame()
        issue_roots = fact_issue_signals.select(['mention_id', 'issue_category']).join(mention_roots.select(['mention_id', 'root_id']), on='mention_id', how='left') if not fact_issue_signals.is_empty() else pl.DataFrame({'mention_id': [], 'issue_category': [], 'root_id': []})
        sentiment_roots = sentiments.join(mention_roots.select(['mention_id', 'root_id']), on='mention_id', how='left') if not sentiments.is_empty() else pl.DataFrame({'mention_id': [], 'sentiment': [], 'root_id': []})
        for row in fact_threads.to_dicts():
            root_id = row['root_id']
            thread_mentions = mention_roots.filter(pl.col('root_id') == root_id) if not mention_roots.is_empty() else pl.DataFrame()
            author_diversity = _ratio(int(thread_mentions.select(pl.col('author_id').n_unique()).item()) if not thread_mentions.is_empty() else 0, int(row['total_mentions']))
            thread_sentiments = sentiment_roots.filter(pl.col('root_id') == root_id) if not sentiment_roots.is_empty() else pl.DataFrame()
            positive = int(thread_sentiments.filter(pl.col('sentiment') == 'positive').height) if not thread_sentiments.is_empty() else 0
            negative = int(thread_sentiments.filter(pl.col('sentiment') == 'negative').height) if not thread_sentiments.is_empty() else 0
            mixed_sentiment = min(positive, negative) / max(positive + negative, 1)
            thread_issues = issue_roots.filter(pl.col('root_id') == root_id) if not issue_roots.is_empty() else pl.DataFrame()
            issue_diversity = _ratio(int(thread_issues.select(pl.col('issue_category').n_unique()).item()) if not thread_issues.is_empty() else 0, max(int(thread_issues.height), 1))
            reply_density = _ratio(int(row['reply_count']), max(int(row['comment_count']), 1))
            recursion = _ratio(int(row['max_depth_observed']), max(int(row['total_mentions']), 1))
            controversy_score = round(min(reply_density * 0.28 + recursion * 0.24 + mixed_sentiment * 0.24 + author_diversity * 0.14 + issue_diversity * 0.1, 1.0), 4)
            threads.append(ThreadControversyRow(root_id=root_id, total_mentions=int(row['total_mentions']), controversy_score_proxy=controversy_score, reply_density_proxy=reply_density, recursion_proxy=recursion, mixed_sentiment_proxy=round(mixed_sentiment, 4), author_diversity_proxy=author_diversity, issue_diversity_proxy=issue_diversity, evidence_references=[root_id, *list(row.get('top_comment_ids') or [])]))
        threads.sort(key=lambda item: (-item.controversy_score_proxy, -item.total_mentions, item.root_id))
    return ThreadControversyReport(contract=contract, window=window, threads=threads[:10])

def _build_creator_source_breakdown_report(bundle, contract, window, *, thread_controversy_report):
    fact_mentions = bundle.tables['fact_mentions']
    fact_topics = bundle.tables['fact_topics']
    fact_issue_signals = bundle.tables['fact_issue_signals']
    mention_total = max(int(fact_mentions.height), 1)
    creators = []
    thread_initiators = []
    sources = []
    controversy_threads = {row.root_id for row in thread_controversy_report.threads if row.controversy_score_proxy >= 0.35}
    if not fact_mentions.is_empty():
        topic_frame = _topic_reporting_frame(fact_topics)
        for row in fact_mentions.group_by('author_id').len().rename({'len': 'mention_count'}).sort(['mention_count', 'author_id'], descending=[True, False]).to_dicts():
            creator_id = row['author_id']
            mention_ids = fact_mentions.filter(pl.col('author_id') == creator_id)['mention_id'].to_list()
            creators.append(CreatorBreakdownRow(creator_id=creator_id, mention_count=int(row['mention_count']), conversation_share=_ratio(int(row['mention_count']), mention_total), top_topic_keys=_top_values(topic_frame.filter(pl.col('mention_id').is_in(mention_ids)), 'reporting_topic_key'), top_issue_categories=_top_values(fact_issue_signals.filter(pl.col('mention_id').is_in(mention_ids)), 'issue_category'), controversy_thread_count=sum((1 for root_id in set(fact_mentions.filter(pl.col('author_id') == creator_id)['root_id'].to_list()) if root_id in controversy_threads)), evidence_references=[mention_id for mention_id in mention_ids[:5] if isinstance(mention_id, str)]))
        creators = creators[:10]
        root_mentions = fact_mentions.filter(pl.col('depth') == 0)
        if not root_mentions.is_empty():
            total_threads = max(int(root_mentions.height), 1)
            for row in root_mentions.group_by('author_id').len().rename({'len': 'thread_count'}).sort(['thread_count', 'author_id'], descending=[True, False]).to_dicts():
                thread_initiators.append(ThreadInitiatorRow(initiator_id=row['author_id'], thread_count=int(row['thread_count']), share_of_threads=_ratio(int(row['thread_count']), total_threads)))
            thread_initiators = thread_initiators[:10]
        for row in fact_mentions.group_by('platform').len().rename({'len': 'mention_count'}).sort(['mention_count', 'platform'], descending=[True, False]).to_dicts():
            platform = row['platform']
            mention_ids = fact_mentions.filter(pl.col('platform') == platform)['mention_id'].to_list()
            sources.append(SourceBreakdownRow(source_platform=str(platform), mention_count=int(row['mention_count']), conversation_share=_ratio(int(row['mention_count']), mention_total), top_topic_keys=_top_values(_topic_reporting_frame(fact_topics).filter(pl.col('mention_id').is_in(mention_ids)), 'reporting_topic_key'), top_issue_categories=_top_values(fact_issue_signals.filter(pl.col('mention_id').is_in(mention_ids)), 'issue_category')))
        sources = sources[:10]
    return CreatorSourceBreakdownReport(contract=contract, window=window, creators=creators, thread_initiators=thread_initiators, sources=sources)

def _build_insight_card_bundle(contract, window, *, sov_report, buzz_report, emerging_topics_report, top_issues_report, thread_controversy_report, creator_source_breakdown_report):
    cards = []
    time_window = _window_label(window)
    if sov_report.entities:
        top_entity = sov_report.entities[0]
        if top_entity.delta_mention_count != 0:
            shift_verb = 'gained' if top_entity.delta_mention_count > 0 else 'lost'
            cards.append(InsightCard(insight_type='share_of_voice_shift', title=f'Share of voice shift detected for {top_entity.entity_name or top_entity.canonical_entity_id}', summary=f'{top_entity.entity_name or top_entity.canonical_entity_id} {shift_verb} {abs(top_entity.delta_mention_count)} mention(s) in the later half of the window.', supporting_metrics={'mention_share': top_entity.mention_share, 'delta_mention_share': top_entity.delta_mention_share, 'delta_mention_count': top_entity.delta_mention_count}, evidence_references=top_entity.evidence_references, confidence=min(0.55 + abs(top_entity.delta_mention_share), 0.9), time_window=time_window, filters_used={'canonical_entity_id': top_entity.canonical_entity_id}, source_reports=['sov_report', 'buzz_report']))
    if buzz_report.topic_buzz:
        top_topic = buzz_report.topic_buzz[0]
        cards.append(InsightCard(insight_type='trending_topic', title=f'Topic gaining volume: {top_topic.reporting_topic_label}', summary=f'{top_topic.reporting_topic_label} is reportable in {top_topic.mention_count} raw mentions and {top_topic.effective_mention_count:.2f} effective mentions.', supporting_metrics={'mention_count': top_topic.mention_count, 'effective_mention_count': top_topic.effective_mention_count, 'buzz_score_proxy': top_topic.buzz_score_proxy, 'reportability_score': top_topic.reportability_score or 0.0}, evidence_references=top_topic.evidence_references, confidence=min(0.5 + max(top_topic.growth_ratio_proxy, 0.0) * 0.1 + (top_topic.reportability_score or 0.0) * 0.18, 0.86), time_window=time_window, filters_used={'effective_topic_key': top_topic.reporting_topic_key}, source_reports=['buzz_report']))
        cards.append(InsightCard(insight_type='conversation_hotspot', title=f'Topic buzz concentrating around {top_topic.reporting_topic_label}', summary=f'{top_topic.reporting_topic_label} appears in {top_topic.mention_count} mentions with buzz score {top_topic.buzz_score_proxy}.', supporting_metrics={'mention_count': top_topic.mention_count, 'effective_mention_count': top_topic.effective_mention_count, 'buzz_score_proxy': top_topic.buzz_score_proxy, 'growth_ratio_proxy': top_topic.growth_ratio_proxy}, evidence_references=top_topic.evidence_references, confidence=min(0.52 + max(top_topic.growth_ratio_proxy, 0.0) * 0.12 + (top_topic.reportability_score or 0.0) * 0.16, 0.88), time_window=time_window, filters_used={'reporting_topic_key': top_topic.reporting_topic_key}, source_reports=['buzz_report']))
    if emerging_topics_report.topics:
        topic = emerging_topics_report.topics[0]
        cards.append(InsightCard(insight_type='emerging_topic', title=f'Emerging topic: {topic.reporting_topic_label}', summary=f'{topic.reporting_topic_label} passed quality gates with growth delta {topic.growth_delta or 0:.1f} and reportability score {topic.reportability_score or 0:.2f}.', supporting_metrics={'growth_delta': topic.growth_delta or 0.0, 'quality_score': topic.quality_score or 0.0, 'usefulness_score': topic.usefulness_score or 0.0, 'reportability_score': topic.reportability_score or 0.0}, evidence_references=topic.representative_document_ids[:5], confidence=min(0.54 + (topic.quality_score or 0.0) * 0.14 + (topic.reportability_score or 0.0) * 0.18, 0.92), time_window=time_window, filters_used={'reporting_topic_key': topic.reporting_topic_key}, source_reports=['emerging_topics_report']))
    if top_issues_report.issues:
        issue = top_issues_report.issues[0]
        cards.append(InsightCard(insight_type='issue_warning', title=f'Issue pressure around {issue.issue_category}', summary=f'{issue.issue_category} leads the issue report with pressure score {issue.issue_pressure_proxy} across {issue.mention_count} mentions.', supporting_metrics={'issue_pressure_proxy': issue.issue_pressure_proxy, 'mention_prevalence_ratio': issue.mention_prevalence_ratio, 'growth_delta_mentions': issue.growth_delta_mentions}, evidence_references=issue.evidence_references, confidence=min(0.58 + issue.issue_pressure_proxy / max(issue.issue_pressure_proxy, 1.0) * 0.18, 0.86), time_window=time_window, filters_used={'issue_category': issue.issue_category}, source_reports=['top_issues_report']))
    if thread_controversy_report.threads:
        thread = thread_controversy_report.threads[0]
        cards.append(InsightCard(insight_type='controversy_alert', title=f'Controversy proxy elevated for thread {thread.root_id}', summary=f'Thread {thread.root_id} scored {thread.controversy_score_proxy} on the controversy proxy from reply density, recursion, polarity conflict, and diversity signals.', supporting_metrics={'controversy_score_proxy': thread.controversy_score_proxy, 'reply_density_proxy': thread.reply_density_proxy, 'mixed_sentiment_proxy': thread.mixed_sentiment_proxy}, evidence_references=thread.evidence_references, confidence=min(0.52 + thread.controversy_score_proxy * 0.25, 0.84), time_window=time_window, filters_used={'root_id': thread.root_id}, source_reports=['thread_controversy_report']))
    if creator_source_breakdown_report.creators:
        creator = creator_source_breakdown_report.creators[0]
        cards.append(InsightCard(insight_type='creator_concentration', title=f'Creator concentration around {creator.creator_id}', summary=f'{creator.creator_id} contributed {creator.mention_count} mentions, or {creator.conversation_share:.2%} of the slice.', supporting_metrics={'mention_count': creator.mention_count, 'conversation_share': creator.conversation_share, 'controversy_thread_count': creator.controversy_thread_count}, evidence_references=creator.evidence_references, confidence=min(0.5 + creator.conversation_share * 0.5, 0.8), time_window=time_window, filters_used={'creator_id': creator.creator_id}, source_reports=['creator_source_breakdown_report']))
    return InsightCardBundle(contract=contract, window=window, cards=cards)

def _entity_counts(frame, *, mention_ids):
    if frame.is_empty() or not mention_ids:
        return {}
    filtered = frame.filter(pl.col('mention_id').is_in(list(mention_ids)))
    if filtered.is_empty():
        return {}
    grouped = filtered.group_by('canonical_entity_id').agg(pl.col('mention_id').n_unique().alias('mention_count'), pl.len().alias('entity_hit_count'))
    return {row['canonical_entity_id']: {'mention_count': int(row['mention_count']), 'entity_hit_count': int(row['entity_hit_count'])} for row in grouped.to_dicts()}

def _mention_weights(fact_mentions, weighting_mode):
    if fact_mentions.is_empty():
        return pl.DataFrame({'mention_id': [], 'effective_weight': []})
    column = {'dedup': 'dedup_weight', 'quality': 'quality_weight'}.get(weighting_mode)
    if column is None or column not in fact_mentions.columns:
        return fact_mentions.select('mention_id').with_columns(pl.lit(1.0).alias('effective_weight'))
    return fact_mentions.select(['mention_id', column]).rename({column: 'effective_weight'})

def _effective_mention_volume(fact_mentions, weighting_mode):
    weights = _mention_weights(fact_mentions, weighting_mode)
    if weights.is_empty():
        return 0.0
    return round(float(weights['effective_weight'].sum()), 4)

def _weighted_entity_counts(frame, mention_weights, *, mention_ids=None):
    if frame.is_empty() or mention_weights.is_empty():
        return {}
    working = frame
    if mention_ids:
        working = working.filter(pl.col('mention_id').is_in(list(mention_ids)))
    if working.is_empty():
        return {}
    grouped = working.select(['canonical_entity_id', 'mention_id']).unique().join(mention_weights, on='mention_id', how='left').with_columns(pl.col('effective_weight').fill_null(1.0)).group_by('canonical_entity_id').agg(pl.col('effective_weight').sum().alias('effective_mention_count'))
    return {row['canonical_entity_id']: round(float(row['effective_mention_count']), 4) for row in grouped.to_dicts()}

def _topic_counts(frame, *, mention_ids):
    if frame.is_empty() or not mention_ids:
        return {}
    filtered = frame.filter(pl.col('mention_id').is_in(list(mention_ids)))
    if filtered.is_empty():
        return {}
    grouped = filtered.group_by('reporting_topic_key').agg(pl.col('mention_id').n_unique().alias('mention_count'))
    return {row['reporting_topic_key']: int(row['mention_count']) for row in grouped.to_dicts()}

def _weighted_topic_counts(frame, mention_weights, *, mention_ids=None):
    if frame.is_empty() or mention_weights.is_empty():
        return {}
    working = frame
    if mention_ids:
        working = working.filter(pl.col('mention_id').is_in(list(mention_ids)))
    if working.is_empty():
        return {}
    grouped = working.select(['reporting_topic_key', 'mention_id']).unique().join(mention_weights, on='mention_id', how='left').with_columns(pl.col('effective_weight').fill_null(1.0)).group_by('reporting_topic_key').agg(pl.col('effective_weight').sum().alias('effective_mention_count'))
    return {row['reporting_topic_key']: round(float(row['effective_mention_count']), 4) for row in grouped.to_dicts()}

def _issue_mention_counts(frame):
    if frame.is_empty():
        return {}
    grouped = frame.group_by('issue_category').agg(pl.col('mention_id').n_unique().alias('mention_count'))
    return {row['issue_category']: int(row['mention_count']) for row in grouped.to_dicts()}

def _artifact_map(frame):
    if frame.is_empty():
        return {}
    result = {}
    for row in frame.to_dicts():
        reporting_key = row.get('reporting_topic_key') or row.get('effective_topic_key') or row.get('topic_key')
        if isinstance(reporting_key, str) and reporting_key:
            result[reporting_key] = row
    return result

def _entity_unresolved_reason_breakdown(frame):
    if frame.is_empty() or 'unresolved_reason' not in frame.columns:
        return {}
    unresolved = frame.filter(pl.col('canonical_entity_id').is_null() & pl.col('concept_entity_id').is_null() & pl.col('unresolved_reason').is_not_null())
    if unresolved.is_empty():
        return {}
    grouped = unresolved.group_by('unresolved_reason').agg(pl.col('mention_id').n_unique().alias('mention_count'))
    return {str(row['unresolved_reason']): int(row['mention_count']) for row in grouped.sort(['mention_count', 'unresolved_reason'], descending=[True, False]).to_dicts()}

def _topic_reporting_frame(frame):
    if frame.is_empty():
        return frame
    working = frame.with_columns(pl.coalesce([pl.col('reporting_topic_key'), pl.col('effective_topic_key'), pl.col('topic_key')]).alias('reporting_topic_key'), pl.coalesce([pl.col('reporting_topic_label'), pl.col('effective_topic_label'), pl.col('topic_label')]).alias('reporting_topic_label'))
    if 'reporting_status' in working.columns:
        reportable = working.filter(pl.col('reporting_status') == 'reportable')
        if not reportable.is_empty():
            return reportable
    return working

def _top_values(frame, column, *, limit=3):
    if frame.is_empty() or column not in frame.columns:
        return []
    valid = frame.filter(pl.col(column).is_not_null())
    if valid.is_empty():
        return []
    grouped = valid.group_by(column).len().sort(['len', column], descending=[True, False])
    return [str(row[column]) for row in grouped.head(limit).to_dicts()]

def _issue_pressure_proxy(frame):
    severity_weight = {'low': 0.6, 'medium': 1.0, 'high': 1.45, 'critical_like_proxy': 1.8}
    evidence_weight = {'direct_complaint': 1.0, 'direct_observation': 0.95, 'comparison_based_critique': 0.8, 'question_or_uncertainty': 0.55, 'hearsay_or_rumor': 0.45, 'escalation_signal': 1.15}
    total = 0.0
    working = frame
    for column in ('confidence', 'target_grounding_confidence', 'corroboration_confidence'):
        if column not in working.columns:
            working = working.with_columns(pl.lit(None).alias(column))
    for row in working.select(['severity', 'evidence_mode', 'confidence', 'target_grounding_confidence', 'corroboration_confidence']).to_dicts():
        confidence_multiplier = max(float(row.get('confidence') or 0.0), 0.25)
        grounding_multiplier = max(float(row.get('target_grounding_confidence') or 0.55), 0.35)
        corroboration_multiplier = max(float(row.get('corroboration_confidence') or 0.5), 0.35)
        trust_multiplier = 0.35 + 0.65 * ((confidence_multiplier + grounding_multiplier + corroboration_multiplier) / 3.0)
        total += severity_weight.get(str(row.get('severity')), 1.0) * evidence_weight.get(str(row.get('evidence_mode')), 0.7) * trust_multiplier
    return round(total, 4)

def _optional_float(value):
    if isinstance(value, (float, int)):
        return float(value)
    return None

def _mix(frame, column):
    if frame.is_empty() or column not in frame.columns:
        return {}
    valid = frame.filter(pl.col(column).is_not_null())
    if valid.is_empty():
        return {}
    total = max(int(valid.height), 1)
    grouped = valid.group_by(column).len().sort(column)
    return {str(row[column]): _ratio(int(row['len']), total) for row in grouped.to_dicts()}

def _entity_name(dimensions, entity_id):
    row = dimensions.get(entity_id, {})
    value = row.get('entity_name')
    return str(value) if isinstance(value, str) else None

def _entity_type(dimensions, entity_id):
    row = dimensions.get(entity_id, {})
    value = row.get('entity_type')
    return str(value) if isinstance(value, str) else None

def _head_mentions(frame, *, limit):
    if frame.is_empty() or 'mention_id' not in frame.columns:
        return []
    seen = []
    for mention_id in frame['mention_id'].to_list():
        if isinstance(mention_id, str) and mention_id not in seen:
            seen.append(mention_id)
        if len(seen) >= limit:
            break
    return seen

def _ratio(numerator, denominator):
    if float(denominator) <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)

def _growth_ratio(current, previous):
    if previous <= 0:
        if current <= 0:
            return 0.0
        return round(float(current), 4)
    return round((float(current) - float(previous)) / float(previous), 4)

def _iso(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return None

def _window_label(window):
    if window.analysis_start and window.analysis_end:
        return f'{window.analysis_start} to {window.analysis_end}'
    return 'all_time'
