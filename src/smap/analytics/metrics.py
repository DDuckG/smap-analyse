from __future__ import annotations

import polars as pl

from smap.analytics.metric_contracts import (
    METRICS_SCHEMA_VERSION,
    family_definition_versions,
)
from smap.marts.builder import MartBundle


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _safe_sum(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty() or column not in frame.columns:
        return 0.0
    return round(float(frame[column].fill_null(0.0).sum()), 4)


def _effective_weight_column(weighting_mode: str) -> str:
    if weighting_mode == "quality":
        return "quality_weight"
    if weighting_mode == "dedup":
        return "dedup_weight"
    return "__raw_weight"


def _distribution(frame: pl.DataFrame, column: str, *, denominator: int) -> list[dict[str, object]]:
    if frame.is_empty() or column not in frame.columns:
        return []
    valid = frame.filter(pl.col(column).is_not_null())
    if valid.is_empty():
        return []
    counts = valid.group_by(column).len().sort("len", descending=True)
    return [
        {
            column: row[column],
            "count": int(row["len"]),
            "ratio": _safe_ratio(float(row["len"]), float(denominator)),
        }
        for row in counts.to_dicts()
    ]


def _concentration(frame: pl.DataFrame, column: str) -> dict[str, object]:
    if frame.is_empty() or column not in frame.columns:
        return {"count": 0, "top_share": 0.0, "hhi_proxy": 0.0, "leaders": []}
    valid = frame.filter(pl.col(column).is_not_null())
    if valid.is_empty():
        return {"count": 0, "top_share": 0.0, "hhi_proxy": 0.0, "leaders": []}
    counts = valid.group_by(column).len().sort("len", descending=True)
    total = max(int(counts["len"].sum()), 1)
    shares = [(int(row["len"]) / total) for row in counts.to_dicts()]
    leaders = [
        {
            column: row[column],
            "count": int(row["len"]),
            "ratio": _safe_ratio(float(row["len"]), float(total)),
        }
        for row in counts.head(5).to_dicts()
    ]
    return {
        "count": int(counts.height),
        "top_share": round(max(shares, default=0.0), 4),
        "hhi_proxy": round(sum(share * share for share in shares), 4),
        "leaders": leaders,
    }


def _resolved_mentions(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.filter(pl.col("canonical_entity_id").is_not_null())


def _group_count_rows(
    frame: pl.DataFrame,
    *,
    group_cols: list[str],
    sort_cols: list[str] | None = None,
    descending: list[bool] | None = None,
) -> list[dict[str, object]]:
    if frame.is_empty() or any(column not in frame.columns for column in group_cols):
        return []
    valid = frame.filter(pl.all_horizontal(*(pl.col(column).is_not_null() for column in group_cols)))
    if valid.is_empty():
        return []
    counts = valid.group_by(group_cols).len().rename({"len": "count"})
    if sort_cols:
        counts = counts.sort(sort_cols) if descending is None else counts.sort(sort_cols, descending=descending)
    return counts.to_dicts()


def _business_target_frame(frame: pl.DataFrame) -> pl.DataFrame:
    working = frame
    if "concept_entity_id" not in working.columns:
        working = working.with_columns(pl.lit(None).cast(pl.Utf8).alias("concept_entity_id"))
    if "target_kind" not in working.columns:
        working = working.with_columns(pl.lit(None).cast(pl.Utf8).alias("target_kind"))
    if "target_text" not in working.columns:
        working = working.with_columns(pl.lit(None).cast(pl.Utf8).alias("target_text"))
    if "entity_type" not in working.columns:
        working = working.with_columns(pl.lit(None).cast(pl.Utf8).alias("entity_type"))
    return working.with_columns(
        pl.when(pl.col("canonical_entity_id").is_not_null())
        .then(pl.col("canonical_entity_id"))
        .when(pl.col("concept_entity_id").is_not_null())
        .then(pl.col("concept_entity_id"))
        .otherwise(None)
        .alias("business_target_key"),
        pl.when(pl.col("canonical_entity_id").is_not_null())
        .then(pl.lit("canonical_entity"))
        .when(pl.col("concept_entity_id").is_not_null())
        .then(pl.lit("concept"))
        .otherwise(pl.lit("surface"))
        .alias("business_target_kind"),
    )


def _target_sentiment_distribution(frame: pl.DataFrame) -> list[dict[str, object]]:
    target_cols = [
        "business_target_key",
        "business_target_kind",
        "canonical_entity_id",
        "concept_entity_id",
        "entity_type",
    ]
    required = [*target_cols, "sentiment"]
    if frame.is_empty() or any(column not in frame.columns for column in required):
        return []
    valid = frame.filter(pl.col("business_target_key").is_not_null() & pl.col("sentiment").is_not_null())
    if valid.is_empty():
        return []
    counts = valid.group_by([*target_cols, "sentiment"]).agg(
        pl.len().alias("count"),
        pl.col("target_text").drop_nulls().first().alias("representative_target_text"),
    )
    totals = valid.group_by("business_target_key").len().rename({"len": "target_total"})
    return (
        counts.join(totals, on="business_target_key", how="left")
        .with_columns(
            pl.struct(["count", "target_total"]).map_elements(
                lambda values: _safe_ratio(float(values["count"]), float(values["target_total"])),
                return_dtype=pl.Float64,
            ).alias("ratio")
        )
        .rename({"business_target_key": "target_key", "business_target_kind": "target_kind"})
        .sort(["target_key", "count", "sentiment"], descending=[False, True, False])
        .to_dicts()
    )


def _target_sentiment_distribution_forensic(frame: pl.DataFrame) -> list[dict[str, object]]:
    working = frame
    if "concept_entity_id" not in working.columns:
        working = working.with_columns(pl.lit(None).cast(pl.Utf8).alias("concept_entity_id"))
    target_cols = ["target_key", "target_text", "canonical_entity_id", "concept_entity_id", "entity_type"]
    required = [*target_cols, "sentiment"]
    if working.is_empty() or any(column not in working.columns for column in required):
        return []
    valid = working.filter(pl.col("target_key").is_not_null() & pl.col("sentiment").is_not_null())
    if valid.is_empty():
        return []
    counts = valid.group_by([*target_cols, "sentiment"]).len().rename({"len": "count"})
    totals = valid.group_by("target_key").len().rename({"len": "target_total"})
    return (
        counts.join(totals, on="target_key", how="left")
        .with_columns(
            pl.struct(["count", "target_total"]).map_elements(
                lambda values: _safe_ratio(float(values["count"]), float(values["target_total"])),
                return_dtype=pl.Float64,
            ).alias("ratio")
        )
        .sort(["target_key", "count", "sentiment"], descending=[False, True, False])
        .to_dicts()
    )


def _aspect_target_pair_count(frame: pl.DataFrame) -> list[dict[str, object]]:
    required = ["aspect", "business_target_key", "canonical_entity_id", "concept_entity_id", "business_target_kind"]
    if frame.is_empty() or any(column not in frame.columns for column in required):
        return []
    valid = frame.filter(pl.col("aspect").is_not_null() & pl.col("business_target_key").is_not_null())
    if valid.is_empty():
        return []
    return (
        valid.group_by(
            ["aspect", "business_target_key", "canonical_entity_id", "concept_entity_id", "business_target_kind"]
        )
        .agg(pl.len().alias("count"), pl.col("target_text").drop_nulls().first().alias("representative_target_text"))
        .rename({"business_target_key": "target_key", "business_target_kind": "target_kind"})
        .sort(["aspect", "count", "target_key"], descending=[False, True, False])
        .to_dicts()
    )


def _aspect_target_pair_count_forensic(frame: pl.DataFrame) -> list[dict[str, object]]:
    required = ["aspect", "target_key", "canonical_entity_id"]
    if frame.is_empty() or any(column not in frame.columns for column in required):
        return []
    valid = frame.filter(pl.col("aspect").is_not_null() & pl.col("target_key").is_not_null())
    if valid.is_empty():
        return []
    return (
        valid.group_by(["aspect", "target_key", "canonical_entity_id"])
        .len()
        .rename({"len": "count"})
        .sort(["aspect", "count", "target_key"], descending=[False, True, False])
        .to_dicts()
    )


def _aspect_sentiment_distribution(
    frame: pl.DataFrame,
    *,
    resolved_only: bool,
) -> list[dict[str, object]]:
    required = ["aspect", "sentiment", "canonical_entity_id"]
    if frame.is_empty() or any(column not in frame.columns for column in required):
        return []
    valid = frame.filter(pl.col("aspect").is_not_null() & pl.col("sentiment").is_not_null())
    if resolved_only:
        valid = valid.filter(pl.col("canonical_entity_id").is_not_null())
    if valid.is_empty():
        return []
    counts = valid.group_by(["aspect", "sentiment"]).len().rename({"len": "count"})
    totals = valid.group_by("aspect").len().rename({"len": "aspect_total"})
    return (
        counts.join(totals, on="aspect", how="left")
        .with_columns(
            pl.struct(["count", "aspect_total"]).map_elements(
                lambda values: _safe_ratio(float(values["count"]), float(values["aspect_total"])),
                return_dtype=pl.Float64,
            ).alias("ratio")
        )
        .sort(["aspect", "count", "sentiment"], descending=[False, True, False])
        .to_dicts()
    )


def _issue_rate_by_class(frame: pl.DataFrame, *, mention_volume: int) -> list[dict[str, object]]:
    if frame.is_empty() or "issue_category" not in frame.columns:
        return []
    valid = frame.filter(pl.col("issue_category").is_not_null())
    if valid.is_empty():
        return []
    counts = valid.group_by("issue_category").len().rename({"len": "count"})
    return (
        counts.with_columns(
            pl.lit(mention_volume).alias("mention_total"),
            pl.struct(["count"]).map_elements(
                lambda values: _safe_ratio(float(values["count"]), float(mention_volume)),
                return_dtype=pl.Float64,
            ).alias("ratio"),
        )
        .sort(["count", "issue_category"], descending=[True, False])
        .to_dicts()
    )


def _issue_class_ratio(
    frame: pl.DataFrame,
    *,
    numerator_filter: pl.Expr,
    denominator_column: str,
) -> list[dict[str, object]]:
    required = ["issue_category", denominator_column]
    if frame.is_empty() or any(column not in frame.columns for column in required):
        return []
    denominator_frame = frame.filter(pl.col("issue_category").is_not_null() & pl.col(denominator_column).is_not_null())
    if denominator_frame.is_empty():
        return []
    totals = denominator_frame.group_by("issue_category").len().rename({"len": "issue_total"})
    numerator_frame = denominator_frame.filter(numerator_filter)
    counts = (
        numerator_frame.group_by("issue_category").len().rename({"len": "count"})
        if not numerator_frame.is_empty()
        else totals.select("issue_category").head(0).with_columns(pl.lit(0).cast(pl.Int64).alias("count"))
    )
    return (
        totals.join(counts, on="issue_category", how="left")
        .with_columns(pl.col("count").fill_null(0).cast(pl.Int64))
        .with_columns(
            pl.struct(["count", "issue_total"]).map_elements(
                lambda values: _safe_ratio(float(values["count"]), float(values["issue_total"])),
                return_dtype=pl.Float64,
            ).alias("ratio")
        )
        .sort(["issue_category"], descending=[False])
        .to_dicts()
    )


def _suspicious_author_concentration(frame: pl.DataFrame) -> dict[str, object]:
    if frame.is_empty() or "author_suspicious" not in frame.columns or "author_id" not in frame.columns:
        return {"count": 0, "suspicious_author_share": 0.0, "top_share": 0.0, "leaders": []}
    suspicious = frame.filter(pl.col("author_suspicious") == True)  # noqa: E712
    if suspicious.is_empty():
        return {"count": 0, "suspicious_author_share": 0.0, "top_share": 0.0, "leaders": []}
    concentration = _concentration(suspicious, "author_id")
    return {
        "count": int(suspicious.height),
        "suspicious_author_share": _safe_ratio(float(suspicious.height), float(frame.height)),
        "top_share": concentration["top_share"],
        "leaders": concentration["leaders"],
    }


def build_metrics(bundle: MartBundle, *, weighting_mode: str = "raw") -> dict[str, object]:
    fact_mentions = bundle.tables["fact_mentions"]
    fact_entity_mentions = bundle.tables["fact_entity_mentions"]
    fact_entity_candidate_clusters = bundle.tables.get("fact_entity_candidate_clusters", pl.DataFrame())
    fact_sentiment = bundle.tables["fact_sentiment"]
    fact_target_sentiment = bundle.tables["fact_target_sentiment"]
    fact_aspects = bundle.tables["fact_aspects"]
    fact_threads = bundle.tables["fact_threads"]
    fact_issue_signals = bundle.tables["fact_issue_signals"]

    mention_volume = int(fact_mentions.height)
    working_mentions = (
        fact_mentions.with_columns(pl.lit(1.0).alias("__raw_weight"))
        if not fact_mentions.is_empty()
        else fact_mentions
    )
    weight_column = _effective_weight_column(weighting_mode)
    dedup_weighted_volume = _safe_sum(working_mentions, "dedup_weight") if not working_mentions.is_empty() else 0.0
    quality_weighted_volume = _safe_sum(working_mentions, "quality_weight") if not working_mentions.is_empty() else 0.0
    effective_mention_volume = (
        _safe_sum(working_mentions, weight_column)
        if not working_mentions.is_empty() and weight_column in working_mentions.columns
        else float(mention_volume)
    )

    resolved_entities = _resolved_mentions(fact_entity_mentions)
    concept_mentions = (
        int(fact_entity_mentions.filter(pl.col("concept_entity_id").is_not_null()).select(pl.col("mention_id").n_unique()).item())
        if not fact_entity_mentions.is_empty() and "concept_entity_id" in fact_entity_mentions.columns
        else 0
    )
    resolved_mentions = (
        int(resolved_entities.select(pl.col("mention_id").n_unique()).item()) if not resolved_entities.is_empty() else 0
    )
    total_entity_hits = int(resolved_entities.height)
    reviewable_low_confidence_mentions = 0
    if mention_volume:
        low_confidence_mention_ids: set[str] = set()
        for table in (fact_sentiment, fact_target_sentiment, fact_aspects, fact_issue_signals):
            if table.is_empty() or "confidence" not in table.columns or "mention_id" not in table.columns:
                continue
            low_confidence_mention_ids.update(
                table.filter(pl.col("confidence") < 0.45)["mention_id"].to_list()
            )
        reviewable_low_confidence_mentions = len(low_confidence_mention_ids)

    sov_by_mentions = (
        resolved_entities.group_by("canonical_entity_id")
        .agg(pl.col("mention_id").n_unique().alias("mention_count"))
        .with_columns((pl.col("mention_count") / max(mention_volume, 1)).alias("ratio"))
        .sort("ratio", descending=True)
        .to_dicts()
        if not resolved_entities.is_empty()
        else []
    )
    sov_by_entity_hits = (
        resolved_entities.group_by("canonical_entity_id")
        .len()
        .with_columns((pl.col("len") / max(total_entity_hits, 1)).alias("ratio"))
        .sort("ratio", descending=True)
        .to_dicts()
        if not resolved_entities.is_empty()
        else []
    )
    sov_resolved_mentions_only = (
        resolved_entities.group_by("canonical_entity_id")
        .agg(pl.col("mention_id").n_unique().alias("mention_count"))
        .with_columns((pl.col("mention_count") / max(resolved_mentions, 1)).alias("ratio"))
        .sort("ratio", descending=True)
        .to_dicts()
        if not resolved_entities.is_empty()
        else []
    )

    issue_total = (
        int(fact_issue_signals.filter(pl.col("issue_category").is_not_null()).height)
        if not fact_issue_signals.is_empty() and "issue_category" in fact_issue_signals.columns
        else 0
    )
    root_mentions = (
        fact_mentions.filter(pl.col("depth") == 0)
        if not fact_mentions.is_empty() and "depth" in fact_mentions.columns
        else pl.DataFrame()
    )

    exact_duplicate_mentions = (
        int(working_mentions.filter(pl.col("dedup_kind") == "exact").height)
        if not working_mentions.is_empty() and "dedup_kind" in working_mentions.columns
        else 0
    )
    near_duplicate_mentions = (
        int(working_mentions.filter(pl.col("dedup_kind") == "near").height)
        if not working_mentions.is_empty() and "dedup_kind" in working_mentions.columns
        else 0
    )
    suspicious_mentions = (
        int(working_mentions.filter(pl.col("mention_suspicious") == True).height)  # noqa: E712
        if not working_mentions.is_empty() and "mention_suspicious" in working_mentions.columns
        else 0
    )
    language_rows = _distribution(
        working_mentions,
        "language",
        denominator=max(mention_volume, 1),
    )
    language_source_rows = _distribution(
        working_mentions,
        "language_source",
        denominator=max(mention_volume, 1),
    )
    diagnostics = {
        "mention_volume": mention_volume,
        "raw_mention_volume": mention_volume,
        "effective_mention_volume": effective_mention_volume,
        "dedup_weighted_mention_volume": dedup_weighted_volume,
        "quality_weighted_mention_volume": quality_weighted_volume,
        "active_weighting_mode": weighting_mode,
        "resolved_entity_mention_rate": _safe_ratio(float(resolved_mentions), float(mention_volume)),
        "concept_target_mention_rate": _safe_ratio(float(concept_mentions), float(mention_volume)),
        "reviewable_low_confidence_rate": _safe_ratio(float(reviewable_low_confidence_mentions), float(mention_volume)),
        "batch_candidate_cluster_count": int(fact_entity_candidate_clusters.height) if not fact_entity_candidate_clusters.is_empty() else 0,
        "aspect_fact_total": int(fact_aspects.filter(pl.col("aspect").is_not_null()).height)
        if not fact_aspects.is_empty() and "aspect" in fact_aspects.columns
        else 0,
        "language_distribution": language_rows,
        "language_source_distribution": language_source_rows,
        "explicit_language_ratio": next(
            (
                float(ratio)
                for row in language_source_rows
                if row["language_source"] == "explicit"
                and isinstance((ratio := row.get("ratio")), (float, int))
            ),
            0.0,
        ),
        "inferred_language_ratio": next(
            (
                float(ratio)
                for row in language_source_rows
                if row["language_source"] == "inferred"
                and isinstance((ratio := row.get("ratio")), (float, int))
            ),
            0.0,
        ),
        "duplicate_rate": _safe_ratio(float(exact_duplicate_mentions + near_duplicate_mentions), float(mention_volume)),
        "exact_duplicate_rate": _safe_ratio(float(exact_duplicate_mentions), float(mention_volume)),
        "near_duplicate_rate": _safe_ratio(float(near_duplicate_mentions), float(mention_volume)),
        "spam_suspicion_rate": _safe_ratio(float(suspicious_mentions), float(mention_volume)),
        "suspicious_author_concentration": _suspicious_author_concentration(working_mentions),
    }
    presence_attention = {
        "sov_by_mentions": sov_by_mentions,
        "sov_by_entity_hits": sov_by_entity_hits,
        "sov_resolved_mentions_only": sov_resolved_mentions_only,
        "author_concentration": _concentration(fact_mentions, "author_id"),
        "thread_initiator_concentration": _concentration(root_mentions, "author_id"),
        "channel_concentration": _concentration(fact_mentions, "platform"),
    }
    business_target_sentiment = _business_target_frame(fact_target_sentiment) if not fact_target_sentiment.is_empty() else fact_target_sentiment
    business_aspects = _business_target_frame(fact_aspects) if not fact_aspects.is_empty() else fact_aspects
    perception = {
        "mention_sentiment_distribution": _distribution(
            fact_sentiment,
            "sentiment",
            denominator=max(
                int(fact_sentiment.filter(pl.col("sentiment").is_not_null()).height)
                if not fact_sentiment.is_empty() and "sentiment" in fact_sentiment.columns
                else 0,
                1,
            ),
        ),
        "target_sentiment_distribution_global": _distribution(
            fact_target_sentiment,
            "sentiment",
            denominator=max(
                int(fact_target_sentiment.filter(pl.col("sentiment").is_not_null()).height)
                if not fact_target_sentiment.is_empty() and "sentiment" in fact_target_sentiment.columns
                else 0,
                1,
            ),
        ),
        "target_sentiment_distribution": _target_sentiment_distribution(business_target_sentiment),
        "target_sentiment_distribution_forensic": _target_sentiment_distribution_forensic(fact_target_sentiment),
        "aspect_occurrence_count": _group_count_rows(
            fact_aspects,
            group_cols=["aspect"],
            sort_cols=["count", "aspect"],
            descending=[True, False],
        ),
        "aspect_target_pair_count": _aspect_target_pair_count(business_aspects),
        "aspect_target_pair_count_forensic": _aspect_target_pair_count_forensic(fact_aspects),
        "aspect_sentiment_distribution": _aspect_sentiment_distribution(fact_aspects, resolved_only=False),
        "aspect_sentiment_resolved_only": _aspect_sentiment_distribution(fact_aspects, resolved_only=True),
        "issue_signal_rate_global": _safe_ratio(float(issue_total), float(mention_volume)),
        "issue_signal_rate": _issue_rate_by_class(fact_issue_signals, mention_volume=mention_volume),
        "issue_confirmed_rate": _issue_class_ratio(
            fact_issue_signals,
            numerator_filter=pl.col("evidence_mode").is_in(["direct_complaint", "direct_observation"]),
            denominator_column="evidence_mode",
        ),
        "issue_direct_complaint_rate": _issue_class_ratio(
            fact_issue_signals,
            numerator_filter=pl.col("evidence_mode") == "direct_complaint",
            denominator_column="evidence_mode",
        ),
        "issue_uncertain_rate": _issue_class_ratio(
            fact_issue_signals,
            numerator_filter=pl.col("evidence_mode").is_in(["question_or_uncertainty", "hearsay_or_rumor"]),
            denominator_column="evidence_mode",
        ),
        "severe_issue_rate": _issue_class_ratio(
            fact_issue_signals,
            numerator_filter=pl.col("severity").is_in(["high", "critical_like_proxy"]),
            denominator_column="severity",
        ),
    }
    interaction_structure = {
        "reply_to_comment_ratio_proxy": _safe_ratio(
            float(fact_threads["reply_count"].sum()) if not fact_threads.is_empty() else 0.0,
            float(fact_threads["comment_count"].sum()) if not fact_threads.is_empty() else 0.0,
        ),
        "interaction_recursion_proxy": _safe_ratio(
            float(fact_threads["max_depth_observed"].sum()) if not fact_threads.is_empty() else 0.0,
            float(fact_threads["total_mentions"].sum()) if not fact_threads.is_empty() else 0.0,
        ),
    }
    return {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "metric_definition_versions": family_definition_versions(),
        "diagnostics": diagnostics,
        "presence_attention": presence_attention,
        "perception": perception,
        "interaction_structure": interaction_structure,
    }
