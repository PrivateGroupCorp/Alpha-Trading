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
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
