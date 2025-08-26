"""Command line interface for alpha trading utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from alpha.core.io import read_mt_tsv, to_parquet
from alpha.core.indicators import atr, is_doji_series, true_range
from alpha.core.validate import validate_ohlc_frame


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
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
