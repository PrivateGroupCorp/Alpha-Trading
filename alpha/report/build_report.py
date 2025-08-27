from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .utils import load_csv, load_json, relpath


@dataclass
class ReportCfg:
    title_prefix: str = "Alpha-POI Report"
    logo_path: str | None = None
    theme: str = "light"
    last_n_bars_default: int = 500
    date_format: str = "%Y-%m-%d %H:%M"
    number_format: Dict[str, Any] | None = None
    sections: Dict[str, bool] | None = None
    tables: Dict[str, Any] | None = None
    charts: Dict[str, Any] | None = None
    notes: Dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Artifact collection
# ---------------------------------------------------------------------------

def collect_artifacts(base_dir: str, symbol: str, tf: str) -> Dict[str, pd.Path]:
    """Scan artifacts directory for standard files."""

    from pathlib import Path

    base = Path(base_dir)
    paths: Dict[str, Path] = {}

    token = f"{symbol}_{tf}"

    # Structure
    s_dir = base / "structure" / token
    if s_dir.exists():
        pngs = sorted(s_dir.glob("structure_last*.png"))
        if pngs:
            paths["structure_png"] = pngs[0]
        t_timeline = s_dir / "trend_timeline.csv"
        if t_timeline.exists():
            paths["trend_timeline"] = t_timeline
        t_summary = s_dir / "trend_summary.json"
        if t_summary.exists():
            paths["trend_summary"] = t_summary

    # POI
    p_dir = base / "poi" / token
    if p_dir.exists():
        pngs = sorted(p_dir.glob("poi_last*.png"))
        if pngs:
            paths["poi_png"] = pngs[0]
        z_csv = p_dir / "poi_zones.csv"
        if z_csv.exists():
            paths["poi_zones"] = z_csv
        z_sum = p_dir / "poi_summary.json"
        if z_sum.exists():
            paths["poi_summary"] = z_sum

    # Liquidity
    l_dir = base / "liquidity" / token
    if l_dir.exists():
        asia_csv = l_dir / "asia_range_daily.csv"
        if asia_csv.exists():
            paths["asia_range_daily"] = asia_csv
        eq_csv = l_dir / "eq_clusters.csv"
        if eq_csv.exists():
            paths["eq_clusters"] = eq_csv
        sw_csv = l_dir / "sweeps.csv"
        if sw_csv.exists():
            paths["sweeps"] = sw_csv

    # Execution
    e_dir = base / "execution" / token
    if e_dir.exists():
        t_csv = e_dir / "trades.csv"
        if t_csv.exists():
            paths["trades"] = t_csv
        t_sum = e_dir / "trades_summary.json"
        if t_sum.exists():
            paths["trades_summary"] = t_sum
        eq_curve = e_dir / "equity_curve.csv"
        if eq_curve.exists():
            paths["equity_curve"] = eq_curve

    # Backtest
    b_dir = base / "backtest" / token
    if b_dir.exists():
        bt_sum = b_dir / "bt_summary.json"
        if bt_sum.exists():
            paths["bt_summary"] = bt_sum
        fills = b_dir / "fills.csv"
        if fills.exists():
            paths["fills"] = fills

    return paths


# ---------------------------------------------------------------------------
# Metrics loading
# ---------------------------------------------------------------------------

def load_metrics_and_tables(paths: Dict[str, pd.Path]) -> Dict[str, Any]:
    """Load JSON/CSV metrics and build DataFrame previews."""

    kpi: Dict[str, Any] = {}
    tables: Dict[str, pd.DataFrame] = {}

    if p := paths.get("poi_zones"):
        df = load_csv(p)
        tables["poi_zones"] = df
        if not df.empty:
            kpi["poi"] = {
                "n_zones": int(len(df)),
                "median_width_pips": float(df.get("width_pips", pd.Series()).median(skipna=True)),
            }

    if p := paths.get("asia_range_daily"):
        df = load_csv(p)
        tables["asia_range_daily"] = df
        if not df.empty:
            kpi.setdefault("liquidity", {})["asia_break_up_share"] = float(
                df.get("break_up", pd.Series()).mean(skipna=True)
            )
            kpi["liquidity"]["median_eq_width_pips"] = float(
                df.get("width_pips", pd.Series()).median(skipna=True)
            )

    if p := paths.get("trend_summary"):
        data = load_json(p)
        if data:
            kpi["structure"] = {
                "share_up": data.get("share_time_up"),
                "share_down": data.get("share_time_down"),
                "share_range": data.get("share_time_range"),
                "n_runs": data.get("n_reversals"),
            }

    if p := paths.get("trades_summary"):
        data = load_json(p)
        if data:
            kpi["trades"] = {
                "n": data.get("n_trades"),
                "win_rate": data.get("win_rate"),
                "avg_R": data.get("avg_R"),
                "maxDD_R": data.get("maxDD_R"),
            }

    if p := paths.get("trades"):
        df = load_csv(p)
        if not df.empty:
            df = df.sort_values(by=df.columns[0])
            tables["trades"] = df

    if p := paths.get("eq_clusters"):
        df = load_csv(p)
        if not df.empty:
            tables["eq_clusters"] = df

    if p := paths.get("sweeps"):
        df = load_csv(p)
        if not df.empty:
            tables["sweeps"] = df

    return kpi, tables


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_html(
    cfg: ReportCfg,
    paths: Dict[str, pd.Path],
    kpi: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    out_html_path: str,
    assets_dir: Optional[str] = None,
) -> None:
    from pathlib import Path

    out_path = Path(out_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images: Dict[str, str] = {}
    if assets_dir:
        assets = Path(assets_dir)
        assets.mkdir(parents=True, exist_ok=True)
    else:
        assets = None

    for key, p in paths.items():
        if p and p.suffix.lower() == ".png":
            if assets:
                dest = assets / p.name
                shutil.copy(p, dest)
                images[key] = relpath(dest, out_path.parent)
            else:
                images[key] = relpath(p, out_path.parent)

    table_html: Dict[str, str] = {}
    table_links: Dict[str, str] = {}
    for name, df in tables.items():
        table_html[name] = df.to_html(index=False, float_format=lambda x: f"{x:.1f}")
        original = paths.get(name)
        if isinstance(original, Path) and original.exists():
            table_links[name] = relpath(original, out_path.parent)

    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parent / "templates"))
    tmpl = env.get_template("base.html")
    html = tmpl.render(title=cfg.title_prefix, images=images, tables=table_html, links=table_links, kpi=kpi)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(html)


# ---------------------------------------------------------------------------
# Params snapshot
# ---------------------------------------------------------------------------

def snapshot_params(cfg_paths: Dict[str, str], out_path: str) -> None:
    from pathlib import Path
    import yaml

    snapshot: Dict[str, Any] = {}
    for name, path in cfg_paths.items():
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                snapshot[name] = yaml.safe_load(fh) or {}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2)
