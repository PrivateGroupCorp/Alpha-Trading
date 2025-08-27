"""Command line interface for alpha trading utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

from alpha.ops.runner import RunCfg, run_pipeline
from alpha.ops.scheduler import ScheduleCfg, run_once as sched_run_once, start_scheduler
from alpha.ops.audit import run_repo_audit
from alpha.ops.doctor import run_doctor

from alpha.core.io import read_mt_tsv, to_parquet
from alpha.core.indicators import atr, is_doji_series, true_range
from alpha.core.validate import validate_ohlc_frame
from alpha.levels.detector import LevelsCfgFormation, detect_levels_formation
from alpha.levels.proportionality import (
    LevelsCfgProportionality,
    compute_levels_proportionality,
)
from alpha.levels.break_update import (
    LevelsCfgBreakUpdate,
    apply_break_update,
)
from alpha.eval.levels_metrics import compute_levels_metrics, score_levels
from alpha.viz.levels import LevelsVizCfg, build_level_segments, plot_levels
from alpha.structure.swings import SwingsCfg, build_swings, summarize_swings
from alpha.structure.events import EventsCfg, detect_bos_choch, summarize_events
from alpha.structure.quality import QualityCfg, qualify_events, summarize_quality
from alpha.viz.structure import (
    StructureVizCfg,
    build_structure_segments_and_markers,
    plot_structure,
)
from alpha.viz.poi import POIVizCfg, build_poi_layers, plot_poi
from alpha.liquidity.asia import AsiaCfg, asia_range_daily, summarize_asia_ranges
from alpha.qa.health import run_qa
from alpha.qa.utils import load_yaml
from alpha.liquidity.eq_clusters import (
    EqCfg,
    detect_eq_touches,
    build_eq_clusters,
    score_clusters,
    summarize_eq,
)
from alpha.liquidity.sweep import SweepCfg, detect_sweeps, summarize_sweeps
from alpha.poi.ob import PoiCfg, build_poi_zones, build_poi_segments, summarize_poi
from alpha.backtest.vbt_bridge import (
    VBTCfg,
    prepare_context,
    derive_signals,
    run_vectorbt,
)
from alpha.backtest.metrics import summarize_bt
from alpha.backtest.bt_runner import run_backtest_bt
from alpha.report.build_report import (
    ReportCfg,
    collect_artifacts,
    load_metrics_and_tables,
    render_html,
    snapshot_params,
)
from alpha.data.ingest import fetch_and_normalize, save_pipeline


def analyze_levels_data(data: str, symbol: str, tf: str, tz: str, outdir: str) -> None:
    """Run data ingestion and validation for levels pipeline."""

    ohlc = read_mt_tsv(data, symbol=symbol, timeframe=tf, tz=tz)
    meta = validate_ohlc_frame(ohlc)
    meta.update({"symbol": symbol, "timeframe": tf, "tz": tz})

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    to_parquet(ohlc, out_path / "ohlc.parquet")

    meta_path = out_path / "meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def fetch_data_cli(
    provider: str,
    symbol: str,
    tf: str,
    start: str,
    end: str,
    outdir: str,
    profile: str = "default",
    tz: str | None = None,
    merge: bool = True,
    resample_to: str | None = None,
    save_raw: bool | None = None,
) -> None:
    """Fetch, normalize and store OHLC data."""

    result = fetch_and_normalize(provider, symbol, tf, start, end, profile=profile)
    df = result["df"]
    report = result["report"]
    resample_list = [t.strip() for t in resample_to.split(",")] if resample_to else None
    save_pipeline(df, symbol, tf, outdir, merge=merge, resample_to=resample_list, report=report)
    span = report.get("span", [None, None])
    n_rows = report.get("n_rows", 0)
    gaps = report.get("gaps", {}).get("count", 0)
    print(
        f"[fetch-data] provider={provider} symbol={symbol} tf={tf} span={span[0]}..{span[1]} rows={n_rows} gaps={gaps}"
    )


def indicators(
    parquet: str,
    symbol: str,
    tf: str,
    atr_window: int,
    atr_method: str,
    doji_body_ratio: float,
    outdir: str,
) -> None:
    """Compute basic indicators and write them to parquet."""

    df = pd.read_parquet(parquet)

    tr = true_range(df)
    atr_series = atr(df, window=atr_window, method=atr_method)
    doji = is_doji_series(df, body_ratio=doji_body_ratio)

    out_df = pd.DataFrame(
        {
            "tr": tr,
            f"atr_{atr_method}_{atr_window}": atr_series,
            "is_doji": doji,
        }
    )

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path / "indicators.parquet")

    summary = {
        "symbol": symbol,
        "timeframe": tf,
        "atr_min": float(atr_series.min(skipna=True)),
        "atr_max": float(atr_series.max(skipna=True)),
        "atr_mean": float(atr_series.mean(skipna=True)),
        "doji_count": int(doji.sum()),
    }
    with (out_path / "indicators_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"ATR min={summary['atr_min']:.6f} max={summary['atr_max']:.6f} "
        f"mean={summary['atr_mean']:.6f} | doji={summary['doji_count']}"
    )


def analyze_liquidity_asia(
    parquet: str,
    symbol: str,
    tf: str,
    profile: str,
    outdir: str,
) -> None:
    """Extract Asia session ranges and post-session interactions."""

    df = pd.read_parquet(parquet)

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "liquidity.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    profile_cfg = data.get("profiles", {}).get(profile, {})
    asia_session = profile_cfg.get("asia_session", {})
    post_session = profile_cfg.get("post_session", {})
    atr_cfg = profile_cfg.get("atr", {})
    fmt_cfg = profile_cfg.get("formatting", {})

    cfg = AsiaCfg(
        start_h=int(asia_session.get("start_h", AsiaCfg.start_h)),
        end_h=int(asia_session.get("end_h", AsiaCfg.end_h)),
        tz_source=str(asia_session.get("tz_source", AsiaCfg.tz_source)),
        breakout_lookahead_h=int(
            post_session.get("breakout_lookahead_h", AsiaCfg.breakout_lookahead_h)
        ),
        confirm_with_body=bool(post_session.get("confirm_with_body", AsiaCfg.confirm_with_body)),
        atr_window=int(atr_cfg.get("window", AsiaCfg.atr_window)),
        pip_size=float(fmt_cfg.get("pip_size", AsiaCfg.pip_size)),
        tick_size=float(fmt_cfg.get("tick_size", AsiaCfg.tick_size)),
    )

    daily_df = asia_range_daily(df, cfg)
    summary = summarize_asia_ranges(daily_df, cfg)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(out_path / "asia_range_daily.csv", index=False)
    with (out_path / "asia_range_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    spans_df = daily_df[["date", "start_ts", "end_ts"]]
    spans_df.to_csv(out_path / "asia_session_spans.csv", index=False)

    median_width = float(daily_df["width_pips"].median(skipna=True))
    print(
        f"median width(pips)={median_width:.2f} "
        f"break_up_share={summary['break_up_share']:.3f} "
        f"break_down_share={summary['break_down_share']:.3f} "
        f"both_touch_share={summary['both_touch_share']:.3f}"
    )


def analyze_liquidity_eq(
    parquet: str,
    symbol: str,
    tf: str,
    outdir: str,
    profile: str = "h1",
    swings_csv: str | None = None,
    asia_daily_csv: str | None = None,
) -> None:
    """Detect equal-high/equal-low clusters and write artifacts."""

    df = pd.read_parquet(parquet)

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "liquidity.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    global_cfg = data.get("eq_cluster", {})
    profile_cfg = data.get("profiles", {}).get(profile, {})
    profile_eq = profile_cfg.get("eq_cluster", {})
    cfg_dict = {**global_cfg, **profile_eq}
    cfg = EqCfg(**cfg_dict)

    swings_df = pd.read_csv(swings_csv) if swings_csv else None
    if swings_df is not None and "time" in swings_df.columns:
        swings_df["time"] = pd.to_datetime(swings_df["time"], utc=True, errors="coerce")

    asia_df = None
    if asia_daily_csv:
        asia_df = pd.read_csv(asia_daily_csv, parse_dates=["date"])

    touches = detect_eq_touches(df, cfg, swings_df)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    touches.to_csv(out_path / "eq_touches.csv", index=False)

    clusters = build_eq_clusters(df, touches, cfg, asia_df)
    clusters = score_clusters(clusters, cfg)
    clusters.to_csv(out_path / "eq_clusters.csv", index=False)

    segments = clusters[
        [
            "cluster_id",
            "side",
            "price_center",
            "first_time",
            "last_time",
            "price_center",
            "width",
        ]
    ].rename(columns={"price_center": "y"})
    segments.to_csv(out_path / "eq_segments.csv", index=False)

    summary = summarize_eq(clusters, touches, cfg)
    with (out_path / "eq_clusters_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    share_eqh = float((clusters["side"] == "eqh").mean()) if not clusters.empty else 0.0
    share_eql = float((clusters["side"] == "eql").mean()) if not clusters.empty else 0.0
    median_width_pips = (
        float(clusters["width_pips"].median()) if not clusters.empty else float("nan")
    )
    median_score = float(clusters["score"].median()) if not clusters.empty else float("nan")
    print(
        f"touches={len(touches)} "
        f"clusters_valid={len(clusters)} "
        f"share_eqh={share_eqh:.3f} share_eql={share_eql:.3f} "
        f"median_width_pips={median_width_pips:.2f} median_score={median_score:.3f}"
    )


def analyze_liquidity_sweep(
    parquet: str,
    symbol: str,
    tf: str,
    outdir: str,
    profile: str = "h1",
    eq_clusters_csv: str | None = None,
    asia_daily_csv: str | None = None,
    events_csv: str | None = None,
) -> None:
    """Detect liquidity sweeps and write artifacts."""

    df = pd.read_parquet(parquet)

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "liquidity.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    global_cfg = data.get("sweep", {})
    profile_cfg = data.get("profiles", {}).get(profile, {}).get("sweep", {})
    cfg_dict = {**global_cfg, **profile_cfg}
    cfg = SweepCfg(**cfg_dict)

    eq_clusters = (
        pd.read_csv(eq_clusters_csv, parse_dates=["first_time", "last_time"])
        if eq_clusters_csv
        else pd.DataFrame()
    )
    asia_daily = (
        pd.read_csv(asia_daily_csv, parse_dates=["date", "start_ts", "end_ts"])
        if asia_daily_csv
        else pd.DataFrame()
    )
    events_df = pd.read_csv(events_csv) if events_csv else None

    sweeps = detect_sweeps(df, eq_clusters, asia_daily, events_df, cfg)
    summary = summarize_sweeps(sweeps, cfg)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    sweeps.to_csv(out_path / "sweeps.csv", index=False)
    with (out_path / "sweeps_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    if not sweeps.empty:
        y_edge = sweeps.apply(
            lambda r: r["cluster_price_center"] if r["scope"] == "eq_cluster" else r["edge_price"],
            axis=1,
        )
        segments = sweeps[
            ["sweep_id", "scope", "side", "pen_time", "reclaim_time"]
        ].copy()
        segments["y_edge"] = y_edge
        segments.to_csv(out_path / "sweep_segments.csv", index=False)

    grade_counts = sweeps["quality_grade"].value_counts().to_dict()
    link_rate = (
        float(sweeps["linked_event_id"].notna().mean()) if not sweeps.empty else 0.0
    )
    median_pen = (
        float(sweeps["pen_depth_pips"].median()) if not sweeps.empty else float("nan")
    )
    print(
        f"sweeps={len(sweeps)} grade_counts={grade_counts} "
        f"link_rate={link_rate:.2f} median_pen_pips={median_pen:.2f}"
    )


def analyze_levels_formation(parquet: str, symbol: str, tf: str, profile: str, outdir: str) -> None:
    """Detect formation levels from OHLC data and write artifacts."""
    df = pd.read_parquet(parquet)

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "levels.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    profile_cfg = data.get("profiles", {}).get(profile, {})

    cfg = LevelsCfgFormation(
        doji_body_ratio=float(
            profile_cfg.get("doji_body_ratio", LevelsCfgFormation.doji_body_ratio)
        ),
        ignore_doji=bool(profile_cfg.get("ignore_doji", LevelsCfgFormation.ignore_doji)),
        cooldown_bars=int(profile_cfg.get("cooldown_bars", LevelsCfgFormation.cooldown_bars)),
        min_leg_body_ratio=float(
            profile_cfg.get("min_leg_body_ratio", LevelsCfgFormation.min_leg_body_ratio)
        ),
        tick_size=float(profile_cfg.get("tick_size", LevelsCfgFormation.tick_size)),
    )

    levels = detect_levels_formation(df, cfg)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    levels.to_csv(out_path / "levels_formation.csv", index=False)

    summary = {
        "n_bars": int(len(df)),
        "n_levels": int(len(levels)),
        "density_per_1000": float(len(levels) / len(df) * 1000) if len(df) else 0.0,
        "first_time": df.index[0].isoformat() if len(df) else None,
        "last_time": df.index[-1].isoformat() if len(df) else None,
    }
    with (out_path / "levels_formation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"bars={summary['n_bars']}, levels={summary['n_levels']}, "
        f"density_per_1000={summary['density_per_1000']:.2f}"
    )


def analyze_levels_prop(
    parquet: str,
    levels_csv: str,
    symbol: str,
    tf: str,
    profile: str,
    outdir: str,
) -> None:
    """Compute proportionality tags for levels and write artifacts."""

    df = pd.read_parquet(parquet)
    levels = pd.read_csv(levels_csv, parse_dates=["time"])

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "levels.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    profile_cfg = data.get("profiles", {}).get(profile, {})

    cfg = LevelsCfgProportionality(
        proportionality_ratio=float(
            profile_cfg.get("proportionality_ratio", LevelsCfgProportionality.proportionality_ratio)
        ),
        prop_ref_mode=str(profile_cfg.get("prop_ref_mode", LevelsCfgProportionality.prop_ref_mode)),
        atr_window=int(profile_cfg.get("atr_window", LevelsCfgProportionality.atr_window)),
    )

    levels_prop = compute_levels_proportionality(df, levels, cfg)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    levels_prop.to_csv(out_path / "levels_prop.csv", index=False)

    n_levels = len(levels_prop)
    weak_prop_share = float(levels_prop["weak_prop"].mean()) if n_levels else 0.0
    leg_stats = levels_prop["leg_current"] if n_levels else pd.Series(dtype="float64")
    stats = {
        "mean": float(leg_stats.mean()) if n_levels else 0.0,
        "median": float(leg_stats.median()) if n_levels else 0.0,
        "p25": float(leg_stats.quantile(0.25)) if n_levels else 0.0,
        "p75": float(leg_stats.quantile(0.75)) if n_levels else 0.0,
    }
    atr_series = atr(df, window=cfg.atr_window)
    atr_vals = (
        atr_series.iloc[levels_prop["end_idx"].astype(int)]
        if n_levels
        else pd.Series(dtype="float64")
    )
    corr = float(levels_prop["leg_current"].corr(atr_vals)) if n_levels > 1 else 0.0

    summary = {
        "n_levels": int(n_levels),
        "weak_prop_share": weak_prop_share,
        "leg_current_stats": stats,
        "correlation_leg_atr": corr,
    }
    with (out_path / "levels_prop_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"weak_prop_share = {weak_prop_share:.2%}\n"
        f"leg_current mean={stats['mean']:.6f} median={stats['median']:.6f}"
    )


def analyze_levels_break(
    parquet: str,
    levels_csv: str,
    symbol: str,
    tf: str,
    profile: str,
    outdir: str,
) -> None:
    """Determine break/update state for levels and write artifacts."""

    df = pd.read_parquet(parquet)
    levels = pd.read_csv(levels_csv, parse_dates=["time"])

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "levels.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    profile_cfg = data.get("profiles", {}).get(profile, {})
    break_cfg = profile_cfg.get("break_confirm", {})

    cfg = LevelsCfgBreakUpdate(
        break_mode=str(break_cfg.get("mode", LevelsCfgBreakUpdate.break_mode)),
        atr_mult=float(break_cfg.get("atr_mult", LevelsCfgBreakUpdate.atr_mult)),
        pips=float(break_cfg.get("pips", LevelsCfgBreakUpdate.pips)),
        update_on_wick=bool(profile_cfg.get("update_on_wick", LevelsCfgBreakUpdate.update_on_wick)),
        max_update_distance_atr_mult=float(
            profile_cfg.get(
                "max_update_distance_atr_mult",
                LevelsCfgBreakUpdate.max_update_distance_atr_mult,
            )
        ),
        tick_size=float(profile_cfg.get("tick_size", LevelsCfgBreakUpdate.tick_size)),
        pip_size=float(profile_cfg.get("pip_size", LevelsCfgBreakUpdate.pip_size)),
    )

    levels_state = apply_break_update(df, levels, cfg)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    levels_state.to_csv(out_path / "levels_state.csv", index=False)

    n_levels = len(levels_state)
    share_broken = float((levels_state["state"] == "broken").mean()) if n_levels else 0.0
    broken_mask = levels_state["break_idx"] >= 0
    time_to_break = (
        levels_state.loc[broken_mask, "break_idx"] - levels_state.loc[broken_mask, "end_idx"]
    )
    median_ttb = float(time_to_break.median()) if not time_to_break.empty else float("nan")

    print(
        f"levels={n_levels}, share_broken={share_broken:.2%}, "
        f"median_time_to_break={median_ttb:.2f} (bars)"
    )


def analyze_levels_metrics(
    parquet: str,
    levels_state_csv: str,
    symbol: str,
    tf: str,
    eval_profile: str,
    outdir: str,
) -> None:
    """Compute quality metrics and scores for detected levels."""

    df = pd.read_parquet(parquet)
    levels_state = pd.read_csv(levels_state_csv)
    for col in ["time", "break_time", "first_touch_time"]:
        if col in levels_state.columns:
            levels_state[col] = pd.to_datetime(levels_state[col], utc=True, errors="coerce")

    metrics_row_df, per_level_df = compute_levels_metrics(df, levels_state)

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "eval.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        eval_cfg = yaml.safe_load(fh) or {}

    scores = score_levels(metrics_row_df.iloc[0], tf=eval_profile, eval_cfg=eval_cfg)
    metrics_row_df = metrics_row_df.assign(**scores)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_row_df.to_csv(out_path / "levels_metrics.csv", index=False)
    with (out_path / "levels_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_row_df.iloc[0].to_dict(), fh, indent=2)
    if not per_level_df.empty:
        per_level_df.to_csv(out_path / "levels_per_level_metrics.csv", index=False)

    row = metrics_row_df.iloc[0]
    print(
        "density={:.2f} share_broken={:.2%} weak_prop_share={:.2%} "
        "ttb_median={:.2f} score_total={:.2f}".format(
            row.get("density_per_1000", float("nan")),
            row.get("share_broken", float("nan")),
            row.get("weak_prop_share", float("nan")),
            row.get("ttb_median", float("nan")),
            row.get("score_total", float("nan")),
        )
    )


def analyze_levels_viz(
    parquet: str,
    levels_csv: str,
    symbol: str,
    tf: str,
    profile: str,
    outdir: str,
    last_n_bars: int = 500,
    full: bool = False,
) -> None:
    """Plot detected levels on price chart and write artifacts."""

    df = pd.read_parquet(parquet)
    levels_df = pd.read_csv(levels_csv)
    for col in ["time", "break_time", "first_touch_time"]:
        if col in levels_df.columns:
            levels_df[col] = pd.to_datetime(levels_df[col], utc=True, errors="coerce")

    segments_df, markers_df = build_level_segments(
        df, levels_df, window_last_n=last_n_bars if last_n_bars else None
    )
    tail_df = df.tail(last_n_bars) if last_n_bars else df

    viz_cfg_path = Path(__file__).resolve().parents[1] / "config" / "viz.yml"
    cfg_dict = {}
    if viz_cfg_path.exists():
        with viz_cfg_path.open("r", encoding="utf-8") as fh:
            cfg_dict = (yaml.safe_load(fh) or {}).get("levels_plot", {})
    cfg = LevelsVizCfg(**cfg_dict)

    levels_cfg_path = Path(__file__).resolve().parents[1] / "config" / "levels.yml"
    tick_size = cfg.tick_size
    if levels_cfg_path.exists():
        with levels_cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        profile_cfg = data.get("profiles", {}).get(profile, {})
        tick_size = float(profile_cfg.get("tick_size", tick_size))
    cfg.tick_size = tick_size

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    suffix = f"last{last_n_bars}" if last_n_bars else "full"
    title = f"{symbol} {tf} — Levels ({'last ' + str(last_n_bars) if last_n_bars else 'full'})"

    plot_levels(
        tail_df,
        segments_df,
        markers_df,
        cfg,
        str(out_path / f"levels_{suffix}.png"),
        title,
    )

    segments_df.to_csv(out_path / f"levels_segments_{suffix}.csv", index=False)
    markers_df.to_csv(out_path / f"levels_markers_{suffix}.csv", index=False)

    if full and last_n_bars:
        seg_full, mark_full = build_level_segments(df, levels_df, window_last_n=None)
        plot_levels(
            df,
            seg_full,
            mark_full,
            cfg,
            str(out_path / "levels_full.png"),
            f"{symbol} {tf} — Levels (full)",
        )
        seg_full.to_csv(out_path / "levels_segments_full.csv", index=False)
        mark_full.to_csv(out_path / "levels_markers_full.csv", index=False)


def analyze_structure_swings(
    parquet: str,
    levels_csv: str,
    symbol: str,
    tf: str,
    profile: str,
    outdir: str,
) -> None:
    """Build structure swings and write artifacts."""

    df = pd.read_parquet(parquet)
    levels_df = pd.read_csv(levels_csv)
    if "time" in levels_df.columns:
        levels_df["time"] = pd.to_datetime(levels_df["time"], utc=True, errors="coerce")

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "structure.yml"
    cfg_dict = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg_dict = yaml.safe_load(fh) or {}
    profile_cfg = cfg_dict.get("profiles", {}).get(profile, {})

    cfg = SwingsCfg(
        use_only_strong_swings=bool(
            profile_cfg.get("use_only_strong_swings", SwingsCfg.use_only_strong_swings)
        ),
        swing_merge_atr_mult=float(
            profile_cfg.get("swing_merge_atr_mult", SwingsCfg.swing_merge_atr_mult)
        ),
        min_gap_bars=int(profile_cfg.get("min_gap_bars", SwingsCfg.min_gap_bars)),
        min_price_delta_atr_mult=float(
            profile_cfg.get("min_price_delta_atr_mult", SwingsCfg.min_price_delta_atr_mult)
        ),
        keep_latest_on_tie=bool(
            profile_cfg.get("keep_latest_on_tie", SwingsCfg.keep_latest_on_tie)
        ),
        atr_window=int(profile_cfg.get("atr_window", SwingsCfg.atr_window)),
    )

    swings_df = build_swings(df, levels_df, cfg)
    summary = summarize_swings(swings_df, levels_df, cfg)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    swings_out = swings_df.copy()
    swings_out["src_level_ids"] = swings_out["src_level_ids"].apply(
        lambda ids: ",".join(map(str, ids))
    )
    swings_out.to_csv(out_path / "swings.csv", index=False)

    with (out_path / "swings_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"levels_in={summary['n_levels_in']}, levels_used={summary['n_levels_used']}, "
        f"swings={summary['n_swings']}, merge_same={summary['merge_same_type_count']}, "
        f"merge_nearby={summary['merge_nearby_count']}"
    )


def analyze_structure_events(
    parquet: str,
    swings_csv: str,
    symbol: str,
    tf: str,
    profile: str,
    outdir: str,
) -> None:
    """Detect structure events (BOS/CHoCH) and write artifacts."""

    df = pd.read_parquet(parquet)
    swings_df = pd.read_csv(swings_csv)
    if "time" in swings_df.columns:
        swings_df["time"] = pd.to_datetime(swings_df["time"], utc=True, errors="coerce")

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "structure.yml"
    cfg_dict = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg_dict = yaml.safe_load(fh) or {}
    profile_cfg = cfg_dict.get("profiles", {}).get(profile, {})

    cfg = EventsCfg(
        bos_break_mult_atr=float(
            profile_cfg.get("bos_break_mult_atr", EventsCfg.bos_break_mult_atr)
        ),
        bos_leg_min_atr_mult=float(
            profile_cfg.get("bos_leg_min_atr_mult", EventsCfg.bos_leg_min_atr_mult)
        ),
        atr_window=int(profile_cfg.get("atr_window", EventsCfg.atr_window)),
        event_cooldown_bars=int(
            profile_cfg.get("event_cooldown_bars", EventsCfg.event_cooldown_bars)
        ),
        confirm_with_body_only=bool(
            profile_cfg.get("confirm_with_body_only", EventsCfg.confirm_with_body_only)
        ),
    )

    events_df = detect_bos_choch(df, swings_df, cfg)
    summary = summarize_events(events_df, cfg)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(out_path / "events.csv", index=False)
    with (out_path / "events_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"BOS={summary['n_bos']}, CHoCH={summary['n_choch']}, "
        f"up/down=({summary['up']}/{summary['down']}), "
        f"median_margin_norm={summary['median_break_margin_norm']:.3f}"
    )


def analyze_structure_quality(
    parquet: str,
    events_csv: str,
    symbol: str,
    tf: str,
    profile: str,
    outdir: str,
) -> None:
    """Qualify structure events and write quality artifacts."""

    df = pd.read_parquet(parquet)
    events_df = pd.read_csv(events_csv, parse_dates=["time"])

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "structure.yml"
    cfg_dict = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg_dict = yaml.safe_load(fh) or {}
    profile_cfg = cfg_dict.get("profiles", {}).get(profile, {})

    cfg = QualityCfg(
        min_break_margin_norm=float(
            profile_cfg.get("min_break_margin_norm", QualityCfg.min_break_margin_norm)
        ),
        ft_bars_min=int(profile_cfg.get("ft_bars_min", QualityCfg.ft_bars_min)),
        ft_backslide_tol_mult=float(
            profile_cfg.get("ft_backslide_tol_mult", QualityCfg.ft_backslide_tol_mult)
        ),
        ft_distance_atr_mult=float(
            profile_cfg.get("ft_distance_atr_mult", QualityCfg.ft_distance_atr_mult)
        ),
        retest_window_bars=int(
            profile_cfg.get("retest_window_bars", QualityCfg.retest_window_bars)
        ),
        retest_tol_atr_mult=float(
            profile_cfg.get("retest_tol_atr_mult", QualityCfg.retest_tol_atr_mult)
        ),
        sweep_window_bars=int(profile_cfg.get("sweep_window_bars", QualityCfg.sweep_window_bars)),
        sweep_tol_atr_mult=float(
            profile_cfg.get("sweep_tol_atr_mult", QualityCfg.sweep_tol_atr_mult)
        ),
        atr_window=int(profile_cfg.get("atr_window", QualityCfg.atr_window)),
        quality_weights=profile_cfg.get("quality_weights", QualityCfg().quality_weights),
        quality_grades=profile_cfg.get("quality_grades", QualityCfg().quality_grades),
    )

    events_q = qualify_events(df, events_df, cfg)
    summary = summarize_quality(events_q, cfg)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    events_q.to_csv(out_path / "events_qualified.csv", index=False)
    with (out_path / "events_quality_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"share_valid={summary['share_valid']:.2%}, grade_counts={summary['grade_counts']}, "
        f"med_ft_bars={summary['med_ft_bars']:.2f}, retest_rate={summary['retest_rate']:.2%}, "
        f"sweep_rate={summary['sweep_rate']:.2%}"
    )


def analyze_structure_viz(
    parquet: str,
    swings_csv: str,
    events_csv: str,
    symbol: str,
    tf: str,
    profile: str,
    outdir: str,
    last_n_bars: int = 500,
    full: bool = False,
) -> None:
    """Plot swings and structure events on price chart and write artifacts."""

    df = pd.read_parquet(parquet)
    swings_df = pd.read_csv(swings_csv, parse_dates=["time"])
    events_df = pd.read_csv(events_csv, parse_dates=["time"])

    viz_cfg_path = Path(__file__).resolve().parents[1] / "config" / "viz.yml"
    cfg_dict = {}
    if viz_cfg_path.exists():
        with viz_cfg_path.open("r", encoding="utf-8") as fh:
            cfg_dict = (yaml.safe_load(fh) or {}).get("structure_plot", {})
    cfg = StructureVizCfg(**cfg_dict)

    segments_df, markers_df = build_structure_segments_and_markers(
        df,
        swings_df,
        events_df,
        window_last_n=last_n_bars if last_n_bars else None,
        show_only_valid=cfg.show_only_valid,
    )

    tail_df = df.tail(last_n_bars) if last_n_bars else df

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    suffix = f"last{last_n_bars}" if last_n_bars else "full"
    title = f"{symbol} {tf} — Structure ({'last ' + str(last_n_bars) if last_n_bars else 'full'})"

    plot_structure(
        tail_df,
        segments_df,
        markers_df,
        cfg,
        str(out_path / f"structure_{suffix}.png"),
        title,
    )

    segments_df.to_csv(out_path / f"structure_segments_{suffix}.csv", index=False)
    markers_df.to_csv(out_path / f"structure_markers_{suffix}.csv", index=False)

    if full and last_n_bars:
        seg_full, mark_full = build_structure_segments_and_markers(
            df,
            swings_df,
            events_df,
            window_last_n=None,
            show_only_valid=cfg.show_only_valid,
        )
        plot_structure(
            df,
            seg_full,
            mark_full,
            cfg,
            str(out_path / "structure_full.png"),
            f"{symbol} {tf} — Structure (full)",
        )
        seg_full.to_csv(out_path / "structure_segments_full.csv", index=False)
        mark_full.to_csv(out_path / "structure_markers_full.csv", index=False)


# ---- CLI wiring ----

def analyze_poi_ob(
    parquet: str,
    symbol: str,
    tf: str,
    outdir: str,
    profile: str = "h1",
    swings_csv: str = "",
    events_csv: str = "",
    sweeps_csv: str | None = None,
    eq_clusters_csv: str | None = None,
    asia_daily_csv: str | None = None,
    trend_timeline_csv: str | None = None,
) -> None:
    df = pd.read_parquet(parquet)
    swings = pd.read_csv(swings_csv, parse_dates=["time"]) if swings_csv else pd.DataFrame()
    events = pd.read_csv(events_csv, parse_dates=["time"]) if events_csv else pd.DataFrame()
    sweeps = pd.read_csv(sweeps_csv, parse_dates=["pen_time"]) if sweeps_csv else None
    eq_clusters = pd.read_csv(eq_clusters_csv) if eq_clusters_csv else None
    asia_daily = pd.read_csv(asia_daily_csv, parse_dates=["date"]) if asia_daily_csv else None
    trend_timeline = pd.read_csv(trend_timeline_csv, parse_dates=["time"]) if trend_timeline_csv else None

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "poi.yml"
    cfg_data = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg_data = yaml.safe_load(fh) or {}
    poi_cfg = cfg_data.get("poi", {})
    cfg = PoiCfg(
        use_only_valid_events=bool(poi_cfg.get("use_only_valid_events", True)),
        ob_lookback_bars=int(poi_cfg.get("ob", {}).get("lookback_bars", PoiCfg.ob_lookback_bars)),
        ob_max_anchor_gap_bars=int(poi_cfg.get("ob", {}).get("max_anchor_gap_bars", PoiCfg.ob_max_anchor_gap_bars)),
        ob_zone_padding_atr_mult=float(poi_cfg.get("ob", {}).get("zone_padding_atr_mult", PoiCfg.ob_zone_padding_atr_mult)),
        ob_body_ratio_min=float(poi_cfg.get("ob", {}).get("body_ratio_min", PoiCfg.ob_body_ratio_min)),
        ob_prefer_full_wick=bool(poi_cfg.get("ob", {}).get("prefer_full_wick", PoiCfg.ob_prefer_full_wick)),
        fvg_detect=bool(poi_cfg.get("ob", {}).get("fvg_detect", PoiCfg.fvg_detect)),
        merge_overlap=bool(poi_cfg.get("merge", {}).get("overlap", PoiCfg.merge_overlap)),
        merge_gap_atr_mult=float(poi_cfg.get("merge", {}).get("merge_gap_atr_mult", PoiCfg.merge_gap_atr_mult)),
        min_width_atr=float(poi_cfg.get("merge", {}).get("min_width_atr", PoiCfg.min_width_atr)),
        max_width_atr=float(poi_cfg.get("merge", {}).get("max_width_atr", PoiCfg.max_width_atr)),
        min_age_bars=int(poi_cfg.get("merge", {}).get("min_age_bars", PoiCfg.min_age_bars)),
        score_weights=poi_cfg.get("score", {}).get("weights"),
        grade_map=poi_cfg.get("score", {}).get("grade_map"),
        atr_window=int(poi_cfg.get("atr_window", PoiCfg.atr_window)),
        pip_size=float(poi_cfg.get("pip_size", PoiCfg.pip_size)),
        tick_size=float(poi_cfg.get("tick_size", PoiCfg.tick_size)),
    )

    zones = build_poi_zones(
        df,
        swings,
        events,
        sweeps,
        eq_clusters,
        asia_daily,
        trend_timeline,
        cfg,
    )
    segments = build_poi_segments(zones, df)
    summary = summarize_poi(zones, cfg)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    zones.to_csv(out_path / "poi_zones.csv", index=False)
    segments.to_csv(out_path / "poi_segments.csv", index=False)
    with (out_path / "poi_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    med_width = float(zones["width_pips"].median(skipna=True)) if not zones.empty else 0.0
    kind_counts = zones["kind"].value_counts().to_dict()
    top5 = zones.sort_values("score_total", ascending=False).head(5)[
        ["zone_id", "kind", "score_total"]
    ]
    print(
        f"zones={len(zones)} kinds={kind_counts} med_width_pips={med_width:.2f}"\
        f" top5={top5.to_dict('records')}"
    )


def analyze_poi_viz(
    parquet: str,
    zones_csv: str,
    symbol: str,
    tf: str,
    outdir: str,
    profile: str = "h1",
    sweeps_csv: str | None = None,
    eq_clusters_csv: str | None = None,
    asia_daily_csv: str | None = None,
    trend_timeline_csv: str | None = None,
    last_n_bars: int = 500,
    full: bool = False,
) -> None:
    df = pd.read_parquet(parquet)
    zones = pd.read_csv(zones_csv)
    sweeps = pd.read_csv(sweeps_csv, parse_dates=["pen_time"]) if sweeps_csv else None
    eq_clusters = pd.read_csv(eq_clusters_csv) if eq_clusters_csv else None
    asia_daily = (
        pd.read_csv(asia_daily_csv, parse_dates=["start_ts", "end_ts", "date"]) if asia_daily_csv else None
    )
    trend_timeline = (
        pd.read_csv(trend_timeline_csv, parse_dates=["time"]) if trend_timeline_csv else None
    )

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "viz.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg_dict = data.get("poi_plot", {})
    cfg = POIVizCfg(**cfg_dict)

    rects, segments, markers = build_poi_layers(
        df,
        zones,
        sweeps,
        eq_clusters,
        asia_daily,
        trend_timeline,
        cfg,
        window_last_n=last_n_bars,
    )

    tail_df = df.tail(last_n_bars) if last_n_bars and last_n_bars > 0 else df
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    plot_poi(
        tail_df,
        rects,
        segments,
        markers,
        cfg,
        str(out_path / f"poi_last{last_n_bars}.png"),
        f"{symbol} {tf} — POI last{last_n_bars}",
    )

    rects.to_csv(out_path / f"poi_rects_last{last_n_bars}.csv", index=False)
    segments.to_csv(out_path / f"poi_segments_last{last_n_bars}.csv", index=False)
    markers.to_csv(out_path / f"poi_markers_last{last_n_bars}.csv", index=False)

    if full:
        rect_f, seg_f, mark_f = build_poi_layers(
            df,
            zones,
            sweeps,
            eq_clusters,
            asia_daily,
            trend_timeline,
            cfg,
            window_last_n=None,
        )
        plot_poi(
            df,
            rect_f,
            seg_f,
            mark_f,
            cfg,
            str(out_path / "poi_full.png"),
            f"{symbol} {tf} — POI full",
        )


def run_execution_cli(
    m1_parquet: str,
    zones_csv: str,
    symbol: str,
    htf: str,
    outdir: str,
    profile: str = "m1_default",
    trend_timeline_csv: str | None = None,
    sweeps_csv: str | None = None,
    events_csv: str | None = None,
    asia_daily_csv: str | None = None,
    eq_clusters_csv: str | None = None,
    start: str | None = None,
    end: str | None = None,
    initial_equity: float = 10000.0,
) -> None:
    from alpha.execution.engine import ExecCfg, run_execution

    m1_df = pd.read_parquet(m1_parquet)
    zones_df = pd.read_csv(zones_csv)
    trend_timeline = (
        pd.read_csv(trend_timeline_csv, parse_dates=["time"]) if trend_timeline_csv else None
    )
    sweeps_df = pd.read_csv(sweeps_csv) if sweeps_csv else None
    context: Dict[str, pd.DataFrame] = {}
    if events_csv:
        context["events"] = pd.read_csv(events_csv)
    if asia_daily_csv:
        context["asia"] = pd.read_csv(asia_daily_csv)
    if eq_clusters_csv:
        context["eq"] = pd.read_csv(eq_clusters_csv)

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "execution.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    profile_cfg = data.get("profiles", {}).get(profile, {})
    exec_cfg = ExecCfg(
        allow_sessions=profile_cfg.get("allow_sessions", ["EU", "US"]),
        session_hours_utc=profile_cfg.get("session_hours_utc", {}),
        min_minutes_between_trades=profile_cfg.get("min_minutes_between_trades", 5),
        max_concurrent_trades=profile_cfg.get("max_concurrent_trades", 2),
        one_trade_per_zone=profile_cfg.get("one_trade_per_zone", True),
        use_trend_filter=profile_cfg.get("use_trend_filter", False),
        min_zone_grade=profile_cfg.get("min_zone_grade", "B"),
        zone_staleness_max_bars=profile_cfg.get("zone_staleness_max_bars", 60),
        require_nearby_sweep=profile_cfg.get("require_nearby_sweep", False),
        triggers=profile_cfg.get("triggers", {}),
        risk=profile_cfg.get("risk", {}),
        risk_caps=profile_cfg.get("risk_caps", {}),
    )

    start_ts = pd.to_datetime(start, utc=True) if start else None
    end_ts = pd.to_datetime(end, utc=True) if end else None

    result = run_execution(
        m1_df,
        zones_df,
        trend_timeline,
        sweeps_df,
        context,
        exec_cfg,
        start_time=start_ts,
        end_time=end_ts,
        initial_equity_usd=initial_equity,
    )

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    result["trades"].to_csv(out_path / "trades.csv", index=False)
    result["fills"].to_csv(out_path / "fills.csv", index=False)
    result["equity_curve"].to_csv(out_path / "equity_curve.csv", index=False)
    result["summary"].to_json(out_path / "trades_summary.json", orient="records")

    n_trades = int(result["summary"]["n_trades"].iloc[0]) if not result["summary"].empty else 0
    win_rate = float(result["summary"]["win_rate"].iloc[0]) if not result["summary"].empty else 0.0
    avg_R = float(result["summary"]["avg_R"].iloc[0]) if not result["summary"].empty else 0.0
    max_dd = float(result["summary"]["max_dd_R"].iloc[0]) if not result["summary"].empty else 0.0
    print(
        f"n_trades={n_trades} win_rate={win_rate:.2f} avg_R={avg_R:.3f} maxDD_R={max_dd:.2f}"
    )


def run_backtest_bt_cli(
    m1_parquet: str,
    zones_csv: str,
    symbol: str,
    htf: str,
    outdir: str,
    profile: str = "bt_m1",
    trend_timeline_csv: str | None = None,
    sweeps_csv: str | None = None,
    asia_daily_csv: str | None = None,
    eq_clusters_csv: str | None = None,
) -> None:
    summary = run_backtest_bt(
        m1_parquet=m1_parquet,
        zones_csv=zones_csv,
        symbol=symbol,
        htf=htf,
        outdir=outdir,
        profile=profile,
        trend_timeline_csv=trend_timeline_csv,
        sweeps_csv=sweeps_csv,
        asia_daily_csv=asia_daily_csv,
        eq_clusters_csv=eq_clusters_csv,
    )
    print(
        f"n_trades={summary.get('n_trades',0)} win_rate={summary.get('win_rate_trades',0):.2f} "
        f"avg_R={summary.get('avg_R',0):.3f} maxDD_R={summary.get('max_dd_R',0):.2f}"
    )



def run_backtest_vbt(
    m1_parquet: str,
    zones_csv: str,
    symbol: str,
    htf: str,
    outdir: str,
    profile: str = "vbt_m1",
    trend_timeline_csv: str | None = None,
    sweeps_csv: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> None:
    """Run lightweight vectorbt backtest and store artifacts."""

    m1_df = pd.read_parquet(m1_parquet)
    zones_df = pd.read_csv(zones_csv)
    trend_timeline = (
        pd.read_csv(trend_timeline_csv, parse_dates=["time"]) if trend_timeline_csv else None
    )
    sweeps_df = pd.read_csv(sweeps_csv) if sweeps_csv else None

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "backtest.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    profile_cfg = data.get("profiles", {}).get(profile, {})
    cfg = VBTCfg(
        min_zone_grade=profile_cfg.get("min_zone_grade", "B"),
        zone_staleness_max_bars=profile_cfg.get("zone_staleness_max_bars", 180),
        use_trend_filter=profile_cfg.get("use_trend_filter", True),
        require_nearby_sweep=profile_cfg.get("require_nearby_sweep", False),
        touch_reject=profile_cfg.get("triggers", {}).get("touch_reject", {}),
        choch_bos_in_zone=profile_cfg.get("triggers", {}).get("choch_bos_in_zone", {}),
        risk=profile_cfg.get("risk", {}),
        fees=profile_cfg.get("fees", {}),
    )

    start_ts = pd.to_datetime(start, utc=True) if start else None
    end_ts = pd.to_datetime(end, utc=True) if end else None
    if start_ts is not None:
        m1_df = m1_df[m1_df.index >= start_ts]
    if end_ts is not None:
        m1_df = m1_df[m1_df.index <= end_ts]

    m1_ctx = prepare_context(m1_df, zones_df, trend_timeline, sweeps_df, cfg)
    entries_long, entries_short, meta = derive_signals(m1_ctx, cfg)

    initial_equity = float(profile_cfg.get("initial_equity", 10000.0))
    result = run_vectorbt(
        m1_df,
        entries_long,
        entries_short,
        meta,
        cfg,
        initial_equity,
    )
    trades_df = result["trades"]
    equity_df = result["equity_curve"]
    summary = summarize_bt(trades_df, equity_df)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_path / "trades.csv", index=False)
    equity_df.to_csv(out_path / "equity_curve.csv", index=False)
    with (out_path / "bt_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(
        f"trades={summary['n_trades']}(legs={summary['n_legs']}), win_rate_trades={summary['win_rate_trades']:.2%}, avg_R={summary['avg_R']:.2f}, maxDD_R={summary['max_dd_R']:.2f}"
    )
    print(f"by_trigger={summary['by_trigger']}")


def generate_report(
    symbol: str,
    tf: str,
    outdir: str,
    profile: str = "h1",
    last_n_bars: int | None = None,
    title_prefix: str | None = None,
) -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "report.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    report_cfg = ReportCfg(**(data.get("report", {})))

    if title_prefix:
        report_cfg.title_prefix = title_prefix

    base_dir = Path(outdir).resolve().parents[1]
    paths = collect_artifacts(str(base_dir), symbol, tf)
    kpi, tables = load_metrics_and_tables(paths)

    max_rows = report_cfg.tables.get("max_rows_preview", 50) if report_cfg.tables else 50
    tables = {k: df.head(max_rows) for k, df in tables.items()}

    now = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_html = Path(outdir) / f"report_{now}.html"
    assets_dir = str(Path(outdir) / "assets") if not report_cfg.charts or report_cfg.charts.get("embed_png", True) else str(Path(outdir) / "assets")
    if report_cfg.charts and not report_cfg.charts.get("embed_png", True):
        assets_dir = None

    render_html(report_cfg, paths, kpi, tables, str(out_html), assets_dir)

    cfg_base = Path(__file__).resolve().parents[1] / "config"
    cfg_paths = {
        name: str(cfg_base / name)
        for name in [
            "structure.yml",
            "liquidity.yml",
            "poi.yml",
            "execution.yml",
            "backtest.yml",
            "viz.yml",
        ]
    }
    snapshot_params(cfg_paths, str(Path(outdir) / "params_snapshot.json"))

    print(f"Report saved: {out_html}")


def run_pipeline_cli(
    profile: str,
    symbol: str | None = None,
    htf: str | None = None,
    ltf: str | None = None,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    artifacts_root: str | None = None,
    runs_root: str | None = None,
) -> None:
    """Load configuration and execute the pipeline runner."""

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "pipeline.yml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    prof = data.get("profiles", {}).get(profile, {})

    symbol = symbol or prof.get("symbol")
    htf = htf or prof.get("htf")
    ltf = ltf or prof.get("ltf")
    date_cfg = prof.get("date", {})
    start = start or date_cfg.get("start")
    end = end or date_cfg.get("end")
    stages_cfg = prof.get("stages", {})
    behavior_cfg = prof.get("behavior", {}).copy()
    io_cfg = prof.get("io", {}).copy()

    if force:
        behavior_cfg["force"] = True
    if resume:
        behavior_cfg["resume"] = True
    if dry_run:
        behavior_cfg["dry_run"] = True
    if artifacts_root:
        io_cfg["artifacts_root"] = artifacts_root
    if runs_root:
        io_cfg["runs_root"] = runs_root

    run_cfg = RunCfg(
        profile=profile,
        symbol=symbol,
        htf=htf,
        ltf=ltf,
        start=start,
        end=end,
        stages=stages_cfg,
        behavior=behavior_cfg,
        io=io_cfg,
    )

    result = run_pipeline(run_cfg)
    print(json.dumps(result))



def schedule_run(
    profile: str = 'e2e_h1_m1',
    once: bool = False,
    run_now: str | None = None,
    list_schedules: bool = False,
    config_path: str = 'alpha/config/scheduler.yml',
) -> None:
    """Execute scheduler related commands."""
    data = {}
    cfg_path = Path(config_path)
    if cfg_path.exists():
        with cfg_path.open('r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
    all_cfg = {sid: ScheduleCfg(**scfg) for sid, scfg in (data.get('schedules') or {}).items()}
    if list_schedules:
        for sid, scfg in all_cfg.items():
            state = 'enabled' if scfg.enable else 'disabled'
            print(f'{sid}: {state}')
        return
    if run_now:
        scfg = all_cfg[run_now]
        res = sched_run_once(run_now, scfg)
        print(Path(res['scheduler_root']) / 'runs.jsonl')
        print(Path(res['scheduler_root']) / 'last_status.json')
        return
    if once:
        sid = next((s for s, c in all_cfg.items() if c.profile == profile), None)
        if not sid:
            raise SystemExit(f'profile {profile} not found in schedules')
        res = sched_run_once(sid, all_cfg[sid])
        print(Path(res['scheduler_root']) / 'runs.jsonl')
        print(Path(res['scheduler_root']) / 'last_status.json')
        return
    start_scheduler(all_cfg)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha-cli")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("fetch-data")
    p.add_argument("--provider", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--profile", default="default")
    p.add_argument("--tz")
    p.add_argument("--merge", dest="merge", action="store_true", default=True)
    p.add_argument("--no-merge", dest="merge", action="store_false")
    p.add_argument("--resample-to")
    p.add_argument("--save-raw", action="store_true")

    p = sub.add_parser("analyze-levels-data")
    p.add_argument("--data", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--tz", required=True)
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("indicators")
    p.add_argument("--parquet", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--atr-method", choices=["ewm", "wilder"], default="ewm")
    p.add_argument("--doji-body-ratio", type=float, default=0.20)
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("analyze-liquidity-asia")
    p.add_argument("--parquet", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("analyze-liquidity-sweep")
    p.add_argument("--parquet", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--eq-clusters")
    p.add_argument("--asia-daily")
    p.add_argument("--events")

    p = sub.add_parser("generate-report")
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--profile", default="h1")
    p.add_argument("--last-n-bars", type=int)
    p.add_argument("--title-prefix")

    p = sub.add_parser("analyze-levels-formation")
    p.add_argument("--parquet", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("analyze-levels-prop")
    p.add_argument("--parquet", required=True)
    p.add_argument("--levels", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("analyze-levels-break")
    p.add_argument("--parquet", required=True)
    p.add_argument("--levels", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("analyze-levels-metrics")
    p.add_argument("--parquet", required=True)
    p.add_argument("--levels-state", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--eval-profile", required=True)
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("analyze-levels-viz")
    p.add_argument("--parquet", required=True)
    p.add_argument("--levels", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--last-n-bars", type=int, default=500)
    p.add_argument("--full", action="store_true")

    p = sub.add_parser("analyze-structure-swings")
    p.add_argument("--parquet", required=True)
    p.add_argument("--levels", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("analyze-structure-events")
    p.add_argument("--parquet", required=True)
    p.add_argument("--swings", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("analyze-structure-quality")
    p.add_argument("--parquet", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("analyze-structure-viz")
    p.add_argument("--parquet", required=True)
    p.add_argument("--swings", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--last-n-bars", type=int, default=500)

    p.add_argument("--full", action="store_true")
    p = sub.add_parser("analyze-poi-ob")
    p.add_argument("--parquet", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--profile", default="h1")
    p.add_argument("--outdir", required=True)
    p.add_argument("--swings")
    p.add_argument("--events")
    p.add_argument("--sweeps")
    p.add_argument("--eq-clusters")
    p.add_argument("--asia-daily")
    p.add_argument("--trend-timeline")

    p = sub.add_parser("analyze-poi-viz")
    p.add_argument("--parquet", required=True)
    p.add_argument("--zones", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--profile", default="h1")
    p.add_argument("--outdir", required=True)
    p.add_argument("--sweeps")
    p.add_argument("--eq-clusters")
    p.add_argument("--asia-daily")
    p.add_argument("--trend-timeline")
    p.add_argument("--last-n-bars", type=int, default=500)
    p.add_argument("--full", action="store_true")

    p = sub.add_parser("run-execution")
    p.add_argument("--m1-parquet", required=True)
    p.add_argument("--zones", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--htf", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--profile", default="m1_default")
    p.add_argument("--trend-timeline")
    p.add_argument("--sweeps")
    p.add_argument("--events")
    p.add_argument("--asia-daily")
    p.add_argument("--eq-clusters")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--initial-equity", type=float, default=10000.0)

    p = sub.add_parser("run-backtest-vbt")
    p.add_argument("--m1-parquet", required=True)
    p.add_argument("--zones", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--htf", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--profile", default="vbt_m1")
    p.add_argument("--trend-timeline")
    p.add_argument("--sweeps")
    p.add_argument("--start")
    p.add_argument("--end")

    p = sub.add_parser("run-backtest-bt")
    p.add_argument("--m1-parquet", required=True)
    p.add_argument("--zones", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--htf", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--profile", default="bt_m1")
    p.add_argument("--trend-timeline")
    p.add_argument("--sweeps")
    p.add_argument("--asia-daily")
    p.add_argument("--eq-clusters")

    p = sub.add_parser("run-pipeline")
    p.add_argument("--profile", required=True)
    p.add_argument("--symbol")
    p.add_argument("--htf")
    p.add_argument("--ltf")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--artifacts-root")
    p.add_argument("--runs-root")

    p = sub.add_parser("repo-audit")
    p.add_argument("--root", default=".")
    p.add_argument("--outdir", default="artifacts/audit")
    p.add_argument("--deep", action="store_true")

    p = sub.add_parser("project-doctor")
    p.add_argument("--root", default=".")
    p.add_argument("--profile", default="e2e_h1_m1")
    p.add_argument("--apply", dest="apply", action="store_true", default=True)
    p.add_argument("--no-apply", dest="apply", action="store_false")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("schedule-run")
    p.add_argument("--profile", default="e2e_h1_m1")
    p.add_argument("--once", action="store_true")
    p.add_argument("--run-now")
    p.add_argument("--list-schedules", action="store_true")
    p.add_argument("--config", dest="config_path", default="alpha/config/scheduler.yml")

    # QA utilities
    p = sub.add_parser("qa-run")
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--run-id")
    p.add_argument("--artifacts-root", default="artifacts")
    p.add_argument("--strict-mode", choices=["soft", "hard"], default="soft")

    p = sub.add_parser("qa-validate-last")
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--run-id")
    p.add_argument("--artifacts-root", default="artifacts")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "fetch-data":
        fetch_data_cli(
            provider=args.provider,
            symbol=args.symbol,
            tf=args.tf,
            start=args.start,
            end=args.end,
            outdir=args.outdir,
            profile=args.profile,
            tz=args.tz,
            merge=args.merge,
            resample_to=args.resample_to,
            save_raw=args.save_raw,
        )
    elif args.command == "analyze-levels-data":
        analyze_levels_data(
            data=args.data,
            symbol=args.symbol,
            tf=args.tf,
            tz=args.tz,
            outdir=args.outdir,
        )
    elif args.command == "indicators":
        indicators(
            parquet=args.parquet,
            symbol=args.symbol,
            tf=args.tf,
            atr_window=args.atr_window,
            atr_method=args.atr_method,
            doji_body_ratio=args.doji_body_ratio,
            outdir=args.outdir,
        )
    elif args.command == "analyze-liquidity-asia":
        analyze_liquidity_asia(
            parquet=args.parquet,
            symbol=args.symbol,
            tf=args.tf,
            profile=args.profile,
            outdir=args.outdir,
        )
    elif args.command == "analyze-liquidity-sweep":
        analyze_liquidity_sweep(
            parquet=args.parquet,
            symbol=args.symbol,
            tf=args.tf,
            outdir=args.outdir,
            profile=args.profile,
            eq_clusters_csv=args.eq_clusters,
            asia_daily_csv=args.asia_daily,
            events_csv=args.events,
        )
    elif args.command == "analyze-levels-formation":
        analyze_levels_formation(
            parquet=args.parquet,
            symbol=args.symbol,
            tf=args.tf,
            profile=args.profile,
            outdir=args.outdir,
        )
    elif args.command == "analyze-levels-prop":
        analyze_levels_prop(
            parquet=args.parquet,
            levels_csv=args.levels,
            symbol=args.symbol,
            tf=args.tf,
            profile=args.profile,
            outdir=args.outdir,
        )
    elif args.command == "analyze-levels-break":
        analyze_levels_break(
            parquet=args.parquet,
            levels_csv=args.levels,
            symbol=args.symbol,
            tf=args.tf,
            profile=args.profile,
            outdir=args.outdir,
        )
    elif args.command == "analyze-levels-metrics":
        analyze_levels_metrics(
            parquet=args.parquet,
            levels_state_csv=args.levels_state,
            symbol=args.symbol,
            tf=args.tf,
            eval_profile=args.eval_profile,
            outdir=args.outdir,
        )
    elif args.command == "analyze-levels-viz":
        analyze_levels_viz(
            parquet=args.parquet,
            levels_csv=args.levels,
            symbol=args.symbol,
            tf=args.tf,
            profile=args.profile,
            outdir=args.outdir,
            last_n_bars=args.last_n_bars,
            full=args.full,
        )
    elif args.command == "analyze-structure-swings":
        analyze_structure_swings(
            parquet=args.parquet,
            levels_csv=args.levels,
            symbol=args.symbol,
            tf=args.tf,
            profile=args.profile,
            outdir=args.outdir,
        )
    elif args.command == "analyze-structure-events":
        analyze_structure_events(
            parquet=args.parquet,
            swings_csv=args.swings,
            symbol=args.symbol,
            tf=args.tf,
            profile=args.profile,
            outdir=args.outdir,
        )
    elif args.command == "analyze-structure-quality":
        analyze_structure_quality(
            parquet=args.parquet,
            events_csv=args.events,
            symbol=args.symbol,
            tf=args.tf,
            profile=args.profile,
            outdir=args.outdir,
        )
    elif args.command == "analyze-structure-viz":
        analyze_structure_viz(
            parquet=args.parquet,
            swings_csv=args.swings,
            events_csv=args.events,
            symbol=args.symbol,
            tf=args.tf,
            profile=args.profile,
            outdir=args.outdir,
            last_n_bars=args.last_n_bars,
            full=args.full,
        )
    elif args.command == "analyze-poi-ob":
        analyze_poi_ob(
            parquet=args.parquet,
            symbol=args.symbol,
            tf=args.tf,
            outdir=args.outdir,
            profile=args.profile,
            swings_csv=args.swings,
            events_csv=args.events,
            sweeps_csv=args.sweeps,
            eq_clusters_csv=args.eq_clusters,
            asia_daily_csv=args.asia_daily,
            trend_timeline_csv=args.trend_timeline,
        )
    elif args.command == "analyze-poi-viz":
        analyze_poi_viz(
            parquet=args.parquet,
            zones_csv=args.zones,
            symbol=args.symbol,
            tf=args.tf,
            outdir=args.outdir,
            profile=args.profile,
            sweeps_csv=args.sweeps,
            eq_clusters_csv=args.eq_clusters,
            asia_daily_csv=args.asia_daily,
            trend_timeline_csv=args.trend_timeline,
            last_n_bars=args.last_n_bars,
            full=args.full,
        )
    elif args.command == "run-execution":
        run_execution_cli(
            m1_parquet=args.m1_parquet,
            zones_csv=args.zones,
            symbol=args.symbol,
            htf=args.htf,
            outdir=args.outdir,
            profile=args.profile,
            trend_timeline_csv=args.trend_timeline,
            sweeps_csv=args.sweeps,
            events_csv=args.events,
            asia_daily_csv=args.asia_daily,
            eq_clusters_csv=args.eq_clusters,
            start=args.start,
            end=args.end,
            initial_equity=args.initial_equity,
        )
    elif args.command == "run-backtest-bt":
        run_backtest_bt_cli(
            m1_parquet=args.m1_parquet,
            zones_csv=args.zones,
            symbol=args.symbol,
            htf=args.htf,
            outdir=args.outdir,
            profile=args.profile,
            trend_timeline_csv=args.trend_timeline,
            sweeps_csv=args.sweeps,
            asia_daily_csv=args.asia_daily,
            eq_clusters_csv=args.eq_clusters,
        )
    elif args.command == "run-backtest-vbt":
        run_backtest_vbt(
            m1_parquet=args.m1_parquet,
            zones_csv=args.zones,
            symbol=args.symbol,
            htf=args.htf,
            outdir=args.outdir,
            profile=args.profile,
            trend_timeline_csv=args.trend_timeline,
            sweeps_csv=args.sweeps,
            start=args.start,
            end=args.end,
        )
    elif args.command == "run-pipeline":
        run_pipeline_cli(
            profile=args.profile,
            symbol=args.symbol,
            htf=args.htf,
            ltf=args.ltf,
            start=args.start,
            end=args.end,
            force=args.force,
            resume=args.resume,
            dry_run=args.dry_run,
            artifacts_root=args.artifacts_root,
            runs_root=args.runs_root,
        )
    elif args.command == "repo-audit":
        result = run_repo_audit(root=args.root, outdir=args.outdir, deep=args.deep)
        raise SystemExit(result.get("exit_code", 0))
    elif args.command == "project-doctor":
        result = run_doctor(
            root=args.root, profile=args.profile, apply=args.apply, dry_run=args.dry_run
        )
        if not result.get("ok", True):
            raise SystemExit(1)
    elif args.command == "generate-report":
        generate_report(
            symbol=args.symbol,
            tf=args.tf,
            outdir=args.outdir,
            profile=args.profile,
            last_n_bars=args.last_n_bars,
            title_prefix=args.title_prefix,
        )
    elif args.command == "schedule-run":
        schedule_run(
            profile=args.profile,
            once=args.once,
            run_now=args.run_now,
            list_schedules=args.list_schedules,
            config_path=args.config_path,
        )
    elif args.command == "qa-run":
        cfg = load_yaml("alpha/config/qa.yml").get("qa", {})
        result = run_qa(
            symbol=args.symbol,
            tf=args.tf,
            run_id=args.run_id,
            qa_cfg=cfg,
            artifacts_root=args.artifacts_root,
        )
        ok = all(result.gates.values())
        print(f"[QA] {args.symbol} {args.tf} overall={result.overall:.1f}")
        if args.strict_mode == "hard" and not ok:
            raise SystemExit(1)
    elif args.command == "qa-validate-last":
        cfg = load_yaml("alpha/config/qa.yml").get("qa", {})
        result = run_qa(
            symbol=args.symbol,
            tf=args.tf,
            run_id=args.run_id,
            qa_cfg=cfg,
            artifacts_root=args.artifacts_root,
        )
        ok = all(result.gates.values())
        print(f"[QA] {args.symbol} {args.tf} overall={result.overall:.1f}")
        if not ok:
            raise SystemExit(1)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()

