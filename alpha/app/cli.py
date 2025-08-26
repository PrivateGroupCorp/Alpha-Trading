"""Command line interface for alpha trading utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

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


def analyze_levels_formation(
    parquet: str, symbol: str, tf: str, profile: str, outdir: str
) -> None:
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
            profile_cfg.get(
                "proportionality_ratio", LevelsCfgProportionality.proportionality_ratio
            )
        ),
        prop_ref_mode=str(
            profile_cfg.get("prop_ref_mode", LevelsCfgProportionality.prop_ref_mode)
        ),
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
        update_on_wick=bool(
            profile_cfg.get("update_on_wick", LevelsCfgBreakUpdate.update_on_wick)
        ),
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
    share_broken = (
        float((levels_state["state"] == "broken").mean()) if n_levels else 0.0
    )
    broken_mask = levels_state["break_idx"] >= 0
    time_to_break = levels_state.loc[broken_mask, "break_idx"] - levels_state.loc[
        broken_mask, "end_idx"
    ]
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
            profile_cfg.get(
                "min_price_delta_atr_mult", SwingsCfg.min_price_delta_atr_mult
            )
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

# ---- CLI wiring ----

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha-cli")
    sub = parser.add_subparsers(dest="command")

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

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "analyze-levels-data":
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
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
