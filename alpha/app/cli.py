"""Command line interface for alpha trading utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpha.core.io import read_mt_tsv, to_parquet
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
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
