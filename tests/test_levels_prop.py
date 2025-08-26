from pathlib import Path
import sys
import json

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha.core.indicators import atr
from alpha.levels.proportionality import (
    LevelsCfgProportionality,
    compute_levels_proportionality,
)
from alpha.app.cli import (
    analyze_levels_data,
    analyze_levels_formation,
    analyze_levels_prop,
)


def _sample_df():
    idx = pd.date_range("2020", periods=6, freq="H", tz="UTC")
    close = [10.0, 12.0, 13.0, 16.0, 17.0, 18.0]
    open_ = [c - 0.5 for c in close]
    high = [c + 0.2 for c in close]
    low = [c - 0.2 for c in close]
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=idx
    )


def _sample_levels(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time": df.index[1],
                "type": "peak",
                "price": 0.0,
                "start_idx": 0,
                "end_idx": 1,
                "source_closes": "",
                "notes": "",
            },
            {
                "time": df.index[3],
                "type": "peak",
                "price": 0.0,
                "start_idx": 2,
                "end_idx": 3,
                "source_closes": "",
                "notes": "",
            },
            {
                "time": df.index[4],
                "type": "peak",
                "price": 0.0,
                "start_idx": 3,
                "end_idx": 4,
                "source_closes": "",
                "notes": "",
            },
        ]
    )


def test_prev_leg_mode():
    df = _sample_df()
    levels = _sample_levels(df)
    cfg = LevelsCfgProportionality(prop_ref_mode="prev_leg", atr_window=3)
    res = compute_levels_proportionality(df, levels, cfg)
    atr_series = atr(df, window=3)

    assert res["leg_current"].tolist() == [2.0, 3.0, 1.0]
    assert res.loc[0, "leg_ref"] == pytest.approx(atr_series.iloc[1])
    assert res.loc[1, "leg_ref"] == pytest.approx(res.loc[0, "leg_current"])
    assert res.loc[2, "leg_ref"] == pytest.approx(res.loc[1, "leg_current"])
    assert (res["weak_prop"] == (res["leg_current"] < 0.5 * res["leg_ref"])).all()


def test_atr_and_prev_leg_or_atr_modes():
    df = _sample_df()
    levels = _sample_levels(df)
    atr_series = atr(df, window=3)

    cfg_atr = LevelsCfgProportionality(prop_ref_mode="atr", atr_window=3)
    res_atr = compute_levels_proportionality(df, levels, cfg_atr)
    expected_refs = [atr_series.iloc[1], atr_series.iloc[3], atr_series.iloc[4]]
    assert res_atr["leg_ref"].tolist() == pytest.approx(expected_refs)

    cfg_mix = LevelsCfgProportionality(prop_ref_mode="prev_leg_or_atr", atr_window=3)
    res_mix = compute_levels_proportionality(df, levels, cfg_mix)
    assert res_mix.loc[0, "leg_ref"] == pytest.approx(atr_series.iloc[1])
    assert res_mix.loc[1, "leg_ref"] == pytest.approx(
        max(res_mix.loc[0, "leg_current"], atr_series.iloc[3])
    )
    assert res_mix.loc[2, "leg_ref"] == pytest.approx(
        max(res_mix.loc[1, "leg_current"], atr_series.iloc[4])
    )


def test_integration_levels_prop(tmp_path):
    data_dir = tmp_path / "data"
    analyze_levels_data(
        data="data/EURUSD_H1.tsv",
        symbol="EURUSD",
        tf="H1",
        tz="UTC",
        outdir=str(data_dir),
    )
    parquet_path = data_dir / "ohlc.parquet"
    levels_dir = tmp_path / "levels"
    analyze_levels_formation(
        parquet=str(parquet_path),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(levels_dir),
    )
    levels_csv = levels_dir / "levels_formation.csv"
    outdir = tmp_path / "prop"
    analyze_levels_prop(
        parquet=str(parquet_path),
        levels_csv=str(levels_csv),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(outdir),
    )

    csv_path = outdir / "levels_prop.csv"
    assert csv_path.exists()
    levels_prop = pd.read_csv(csv_path, parse_dates=["time"])
    assert len(levels_prop) == len(pd.read_csv(levels_csv))
    assert {"leg_current", "leg_ref", "weak_prop", "prop_ref_mode", "proportionality_ratio"}.issubset(
        levels_prop.columns
    )
    new_cols = [
        "leg_current",
        "leg_ref",
        "weak_prop",
        "prop_ref_mode",
        "proportionality_ratio",
    ]
    assert not levels_prop[new_cols].isna().any().any()

    summary_path = outdir / "levels_prop_summary.json"
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["n_levels"] == len(levels_prop)
    assert 0.0 <= summary["weak_prop_share"] <= 1.0


def test_empty_levels(tmp_path):
    data_dir = tmp_path / "data"
    analyze_levels_data(
        data="data/EURUSD_H1.tsv",
        symbol="EURUSD",
        tf="H1",
        tz="UTC",
        outdir=str(data_dir),
    )
    parquet_path = data_dir / "ohlc.parquet"
    empty_csv = tmp_path / "empty_levels.csv"
    pd.DataFrame(
        columns=[
            "time",
            "type",
            "price",
            "start_idx",
            "end_idx",
            "source_closes",
            "notes",
        ]
    ).to_csv(empty_csv, index=False)
    outdir = tmp_path / "prop"
    analyze_levels_prop(
        parquet=str(parquet_path),
        levels_csv=str(empty_csv),
        symbol="EURUSD",
        tf="H1",
        profile="h1",
        outdir=str(outdir),
    )
    csv_path = outdir / "levels_prop.csv"
    assert csv_path.exists()
    levels_prop = pd.read_csv(csv_path)
    assert levels_prop.empty
    summary_path = outdir / "levels_prop_summary.json"
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["n_levels"] == 0
