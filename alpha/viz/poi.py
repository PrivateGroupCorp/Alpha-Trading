from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class POIVizCfg:
    last_n_bars: int = 500
    dpi: int = 150
    fig_w: float = 13.0
    fig_h: float = 7.0
    price_linewidth: float = 1.0
    alpha_price: float = 0.85

    show_zones: bool = True
    zone_alpha: float = 0.25
    zone_edge_alpha: float = 0.9
    max_zones: int = 40
    min_grade: str = "C"
    draw_zone_labels: bool = True

    show_sweeps: bool = True
    show_eq: bool = True
    show_asia: bool = True
    show_trend: bool = True
    marker_size: int = 54
    eq_band_alpha: float = 0.35
    asia_time_alpha: float = 0.08
    asia_edge_alpha: float = 0.6

    colors: Dict[str, str] = field(
        default_factory=lambda: {
            "price": "#6b7280",
            "demand": "#22c55e",
            "supply": "#ef4444",
            "breaker": "#a855f7",
            "flip": "#f59e0b",
            "sweep_up": "#22c55e",
            "sweep_down": "#ef4444",
            "eqh": "#ef4444",
            "eql": "#0ea5e9",
            "asia_time": "#60a5fa",
            "asia_high": "#60a5fa",
            "asia_low": "#60a5fa",
            "trend_up_bg": "#dcfce7",
            "trend_down_bg": "#fee2e2",
            "trend_range_bg": "#f1f5f9",
        }
    )
    grade_palette: Dict[str, str] = field(
        default_factory=lambda: {
            "A": "#16a34a",
            "B": "#f59e0b",
            "C": "#9ca3af",
        }
    )

    zone_rank_by: str = "score_total"
    pick_top_k: int = 25


RECT_COLS = [
    "zone_id",
    "kind",
    "grade",
    "x_start",
    "x_end",
    "y_bottom",
    "y_top",
    "score",
    "color",
]

SEGMENT_COLS = ["seg_kind", "x_start", "x_end", "y_start", "y_end", "meta"]

MARKER_COLS = ["mark_kind", "time", "y", "label", "ref_id", "color"]


def _tail_window(df: pd.DataFrame, last_n: Optional[int]) -> pd.DataFrame:
    if last_n and last_n > 0:
        return df.tail(last_n)
    return df


def build_poi_layers(
    df: pd.DataFrame,
    zones: pd.DataFrame,
    sweeps: Optional[pd.DataFrame],
    eq_clusters: Optional[pd.DataFrame],
    asia_daily: Optional[pd.DataFrame],
    trend_timeline: Optional[pd.DataFrame],
    cfg: POIVizCfg,
    window_last_n: Optional[int] = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tail_df = _tail_window(df, window_last_n)
    if tail_df.empty:
        return (
            pd.DataFrame(columns=RECT_COLS),
            pd.DataFrame(columns=SEGMENT_COLS),
            pd.DataFrame(columns=MARKER_COLS),
        )
    t0, t1 = tail_df.index[0], tail_df.index[-1]

    zones = zones.copy()
    for col in ["anchor_time", "break_time"]:
        if col in zones.columns:
            zones[col] = pd.to_datetime(zones[col], utc=True, errors="coerce")
    if "break_time" in zones.columns:
        mask = zones["break_time"].between(t0 - pd.Timedelta(days=60), t1)
    else:
        mask = zones["anchor_time"].between(t0 - pd.Timedelta(days=60), t1)
    zones_win = zones[mask].copy()

    if "grade" in zones_win.columns and cfg.min_grade:
        order = ["C", "B", "A"]
        if cfg.min_grade.upper() in order:
            keep = set(order[order.index(cfg.min_grade.upper()) :])
            zones_win = zones_win[zones_win["grade"].str.upper().isin(keep)]

    if cfg.zone_rank_by in zones_win.columns:
        zones_win = zones_win.sort_values(cfg.zone_rank_by, ascending=False)
    zones_win = zones_win.head(min(cfg.pick_top_k, cfg.max_zones))

    rect_rows: list[dict] = []
    if cfg.show_zones and not zones_win.empty:
        for row in zones_win.itertuples(index=False):
            bt = getattr(row, "break_time", pd.NaT)
            at = getattr(row, "anchor_time", pd.NaT)
            x_start = bt if pd.notna(bt) else at
            if pd.isna(x_start):
                continue
            x_start = max(x_start, t0)
            rect_rows.append(
                {
                    "zone_id": getattr(row, "zone_id", getattr(row, "id", -1)),
                    "kind": getattr(row, "kind", ""),
                    "grade": getattr(row, "grade", ""),
                    "x_start": x_start,
                    "x_end": t1,
                    "y_bottom": float(getattr(row, "price_bottom", float("nan"))),
                    "y_top": float(getattr(row, "price_top", float("nan"))),
                    "score": float(getattr(row, cfg.zone_rank_by, float("nan"))),
                    "color": cfg.colors.get(getattr(row, "kind", ""), "#000000"),
                }
            )
    rects_df = pd.DataFrame(rect_rows, columns=RECT_COLS)

    seg_rows: list[dict] = []
    idx = tail_df.index.to_list()
    closes = tail_df["close"].to_list()
    for i in range(len(idx) - 1):
        seg_rows.append(
            {
                "seg_kind": "price",
                "x_start": idx[i],
                "x_end": idx[i + 1],
                "y_start": closes[i],
                "y_end": closes[i + 1],
                "meta": "{}",
            }
        )

    if cfg.show_eq and eq_clusters is not None and not eq_clusters.empty:
        eq = eq_clusters.copy()
        for col in ["first_time", "last_time"]:
            if col in eq.columns:
                eq[col] = pd.to_datetime(eq[col], utc=True, errors="coerce")
        eq = eq[(eq["last_time"] >= t0) & (eq["first_time"] <= t1)]
        for row in eq.itertuples(index=False):
            seg_rows.append(
                {
                    "seg_kind": "eq_band",
                    "x_start": max(getattr(row, "first_time", pd.NaT), t0),
                    "x_end": min(getattr(row, "last_time", pd.NaT), t1),
                    "y_start": float(getattr(row, "price_min", float("nan"))),
                    "y_end": float(getattr(row, "price_max", float("nan"))),
                    "meta": json.dumps(
                        {
                            "cluster_id": getattr(row, "cluster_id", None),
                            "side": getattr(row, "side", ""),
                        }
                    ),
                }
            )

    if cfg.show_asia and asia_daily is not None and not asia_daily.empty:
        asia = asia_daily.copy()
        for col in ["start_ts", "end_ts"]:
            if col in asia.columns:
                asia[col] = pd.to_datetime(asia[col], utc=True, errors="coerce")
        asia = asia[(asia["end_ts"] >= t0) & (asia["start_ts"] <= t1)]
        for row in asia.itertuples(index=False):
            start = max(getattr(row, "start_ts", pd.NaT), t0)
            end = min(getattr(row, "end_ts", pd.NaT), t1)
            seg_rows.append(
                {
                    "seg_kind": "asia_time",
                    "x_start": start,
                    "x_end": end,
                    "y_start": float("nan"),
                    "y_end": float("nan"),
                    "meta": json.dumps({"date": str(getattr(row, "date", ""))}),
                }
            )
            high = float(getattr(row, "asia_high", float("nan")))
            low = float(getattr(row, "asia_low", float("nan")))
            seg_rows.append(
                {
                    "seg_kind": "asia_edge",
                    "x_start": t0,
                    "x_end": t1,
                    "y_start": high,
                    "y_end": high,
                    "meta": json.dumps({"edge": "high", "date": str(getattr(row, "date", ""))}),
                }
            )
            seg_rows.append(
                {
                    "seg_kind": "asia_edge",
                    "x_start": t0,
                    "x_end": t1,
                    "y_start": low,
                    "y_end": low,
                    "meta": json.dumps({"edge": "low", "date": str(getattr(row, "date", ""))}),
                }
            )

    if cfg.show_trend and trend_timeline is not None and not trend_timeline.empty:
        tt = trend_timeline.copy()
        if "time" in tt.columns:
            tt["time"] = pd.to_datetime(tt["time"], utc=True, errors="coerce")
        tt = tt[(tt["time"] >= t0) & (tt["time"] <= t1)]
        tt = tt.sort_values("time")
        times = tt["time"].to_list()
        states = tt.get("state", pd.Series(["range"] * len(tt))).to_list()
        for i in range(len(times) - 1):
            seg_rows.append(
                {
                    "seg_kind": "trend_bg",
                    "x_start": times[i],
                    "x_end": times[i + 1],
                    "y_start": float("nan"),
                    "y_end": float("nan"),
                    "meta": json.dumps({"state": states[i]}),
                }
            )
        if times:
            seg_rows.append(
                {
                    "seg_kind": "trend_bg",
                    "x_start": times[-1],
                    "x_end": t1,
                    "y_start": float("nan"),
                    "y_end": float("nan"),
                    "meta": json.dumps({"state": states[-1]}),
                }
            )

    segments_df = pd.DataFrame(seg_rows, columns=SEGMENT_COLS)

    marker_rows: list[dict] = []
    if cfg.show_sweeps and sweeps is not None and not sweeps.empty:
        sw = sweeps.copy()
        if "pen_time" in sw.columns:
            sw["pen_time"] = pd.to_datetime(sw["pen_time"], utc=True, errors="coerce")
        sw = sw[(sw["pen_time"] >= t0) & (sw["pen_time"] <= t1)]
        for row in sw.itertuples(index=False):
            side = getattr(row, "side", "")
            kind = "sweep_up" if side == "up" else "sweep_down"
            marker_rows.append(
                {
                    "mark_kind": kind,
                    "time": getattr(row, "pen_time"),
                    "y": float(getattr(row, "edge_price", float("nan"))),
                    "label": f"SW {getattr(row, 'quality_grade', '')}".strip(),
                    "ref_id": getattr(row, "sweep_id", None),
                    "color": cfg.colors.get(kind, "#000000"),
                }
            )
    markers_df = pd.DataFrame(marker_rows, columns=MARKER_COLS)

    return rects_df, segments_df, markers_df


def plot_poi(
    df: pd.DataFrame,
    rects_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    markers_df: pd.DataFrame,
    cfg: POIVizCfg,
    out_png_path: str,
    title: str,
) -> None:
    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h), dpi=cfg.dpi)

    # Trend background
    if cfg.show_trend and not segments_df.empty:
        bg = segments_df[segments_df["seg_kind"] == "trend_bg"]
        for seg in bg.itertuples(index=False):
            meta = json.loads(seg.meta) if seg.meta else {}
            state = meta.get("state", "range")
            color = cfg.colors.get(f"trend_{state}_bg", cfg.colors.get("trend_range_bg"))
            ax.axvspan(seg.x_start, seg.x_end, color=color, alpha=1.0, zorder=0)

    ax.plot(
        df.index,
        df["close"],
        color=cfg.colors.get("price", "#6b7280"),
        lw=cfg.price_linewidth,
        alpha=cfg.alpha_price,
        zorder=3,
    )

    if cfg.show_eq and not segments_df.empty:
        eq = segments_df[segments_df["seg_kind"] == "eq_band"]
        for seg in eq.itertuples(index=False):
            meta = json.loads(seg.meta) if seg.meta else {}
            side = meta.get("side", "eqh")
            color = cfg.colors.get(side, "#ef4444")
            ax.fill_between(
                [seg.x_start, seg.x_end],
                seg.y_start,
                seg.y_end,
                color=color,
                alpha=cfg.eq_band_alpha,
                zorder=1,
            )

    if cfg.show_asia and not segments_df.empty:
        asia_time = segments_df[segments_df["seg_kind"] == "asia_time"]
        for seg in asia_time.itertuples(index=False):
            ax.axvspan(
                seg.x_start,
                seg.x_end,
                color=cfg.colors.get("asia_time", "#60a5fa"),
                alpha=cfg.asia_time_alpha,
                zorder=0.5,
            )
        asia_edge = segments_df[segments_df["seg_kind"] == "asia_edge"]
        for seg in asia_edge.itertuples(index=False):
            meta = json.loads(seg.meta) if seg.meta else {}
            edge = meta.get("edge", "high")
            color_key = "asia_high" if edge == "high" else "asia_low"
            color = cfg.colors.get(color_key, cfg.colors.get("asia_time", "#60a5fa"))
            ax.hlines(
                seg.y_start,
                seg.x_start,
                seg.x_end,
                colors=color,
                alpha=cfg.asia_edge_alpha,
                lw=1.0,
                zorder=2,
            )

    if cfg.show_zones and not rects_df.empty:
        for z in rects_df.sort_values("score").itertuples(index=False):
            face = mcolors.to_rgba(cfg.colors.get(z.kind, "#9ca3af"), cfg.zone_alpha)
            edge = mcolors.to_rgba(cfg.colors.get(z.kind, "#9ca3af"), cfg.zone_edge_alpha)
            ax.add_patch(
                plt.Rectangle(
                    (z.x_start, z.y_bottom),
                    (z.x_end - z.x_start),
                    (z.y_top - z.y_bottom),
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.0,
                    zorder=2,
                )
            )
            if cfg.draw_zone_labels:
                mid = (z.y_top + z.y_bottom) / 2
                abbr = {
                    "demand": "D",
                    "supply": "S",
                    "breaker": "B",
                    "flip": "F",
                }.get(z.kind, z.kind[:1].upper())
                label = f"{abbr} {z.grade} {z.score:.2f}".strip()
                ax.text(
                    z.x_start,
                    mid,
                    label,
                    color=cfg.grade_palette.get(z.grade, "black"),
                    fontsize=8,
                    ha="left",
                    va="center",
                )

    if cfg.show_sweeps and not markers_df.empty:
        for m in markers_df.itertuples(index=False):
            marker = "^" if m.mark_kind == "sweep_up" else "v"
            ax.scatter(
                m.time,
                m.y,
                marker=marker,
                color=m.color,
                s=cfg.marker_size,
                zorder=4,
            )
            if m.label:
                va = "bottom" if marker == "^" else "top"
                ax.text(
                    m.time,
                    m.y,
                    m.label,
                    color=m.color,
                    fontsize=6,
                    ha="center",
                    va=va,
                )

    ax.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi)
    plt.close(fig)
