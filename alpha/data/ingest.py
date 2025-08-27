from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from .normalize import NormCfg, normalize_ohlc
from .validators import quality_report
from .resample import resample_ohlc
from .merge import merge_with_existing
from .providers.base import FetchRequest
from .providers.csv_local import CSVLocalProvider


_PROVIDERS = {
    CSVLocalProvider.name: CSVLocalProvider,
}


def _load_config() -> Dict:
    base = Path(__file__).resolve().parents[1] / "config"
    with (base / "data.yml").open("r", encoding="utf-8") as fh:
        data_cfg = yaml.safe_load(fh) or {}
    with (base / "instruments.yml").open("r", encoding="utf-8") as fh:
        inst_cfg = yaml.safe_load(fh) or {}
    return {"data": data_cfg, "instruments": inst_cfg}


def fetch_and_normalize(provider_name, symbol, tf, start, end, profile="default") -> dict:
    cfg = _load_config()
    profile_cfg = cfg["data"].get("profiles", {}).get(profile, {})
    tz = profile_cfg.get("tz", "UTC")

    symbols_cfg = cfg["instruments"].get("symbols", {})
    vendor_symbol = (
        symbols_cfg.get(symbol, {})
        .get("aliases", {})
        .get(provider_name, symbol)
    )

    provider_cls = _PROVIDERS.get(provider_name)
    if provider_cls is None:
        raise ValueError(f"Unknown provider {provider_name}")
    provider_cfg = profile_cfg.get(provider_name, {})

    req = FetchRequest(
        symbol=symbol,
        vendor_symbol=vendor_symbol,
        tf=tf,
        start=start,
        end=end,
        tz=tz,
        chunk_days=int(profile_cfg.get("chunk_days", 7)),
    )
    provider = provider_cls()
    raw_df = provider.fetch(req, **provider_cfg)

    norm_cfg = NormCfg(
        tz=tz,
        drop_incomplete_ohlc=profile_cfg.get("drop_incomplete_ohlc", True),
        fill_volume_nan_with_zero=profile_cfg.get("fill_volume_nan_with_zero", True),
    )
    df = normalize_ohlc(raw_df, norm_cfg)
    report = quality_report(df, tf)
    return {"df": df, "report": report}


def save_pipeline(
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    outdir: str,
    merge: bool = True,
    resample_to: List[str] | None = None,
    report: Dict | None = None,
) -> None:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_path = out_path / "ohlc.parquet"
    df_to_save = merge_with_existing(df, parquet_path) if merge else df
    df_to_save.to_parquet(parquet_path)

    if report is not None:
        with (out_path / "quality_report.json").open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    if resample_to:
        for tf_to in resample_to:
            res_df = resample_ohlc(df_to_save, tf_to)
            res_outdir = out_path.parent / tf_to
            res_outdir.mkdir(parents=True, exist_ok=True)
            res_path = res_outdir / "ohlc.parquet"
            res_df = merge_with_existing(res_df, res_path) if merge else res_df
            res_df.to_parquet(res_path)
