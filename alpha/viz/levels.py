"""Visualization utilities for plotting levels on price charts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import math

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class LevelsVizCfg:
    last_n_bars: int = 500
    linewidth: float = 1.6
    alpha_intact: float = 0.35
    alpha_broken: float = 0.85
    colors: Dict[str, str] = field(
        default_factory=lambda: {
            "peak_intact": "#f59e0b",
            "peak_broken": "#ef4444",
            "trough_intact": "#0ea5e9",
            "trough_broken": "#22c55e",
            "price": "#6b7280",
            "break_vline": "#94a3b8",
        }
    )
    show_break_vlines: bool = True
    show_first_touch_markers: bool = True
    show_labels_on_edges: bool = True
    dpi: int = 150
    fig_w: float = 12.0
    fig_h: float = 6.0
    tick_size: float = 0.0001


SEG_COLS = [
    "type",
    "state",
    "y",
    "t_start",
    "t_end",
    "price_at_plot",
    "break_idx",
    "first_touch_time",
    "update_count",
]

MARKER_COLS = ["kind", "time", "y", "level_type", "state"]


def build_level_segments(
    df: pd.DataFrame,
    levels_df: pd.DataFrame,
    window_last_n: Optional[int] = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build segment and marker data for level visualization."""

    if window_last_n and window_last_n > 0:
        tail_df = df.tail(window_last_n)
    else:
        tail_df = df
    if tail_df.empty:
        return (
            pd.DataFrame(columns=SEG_COLS),
            pd.DataFrame(columns=MARKER_COLS),
        )

    t0, t1 = tail_df.index[0], tail_df.index[-1]

    levels = levels_df.copy()
    for col in ["time", "break_time", "first_touch_time"]:
        if col in levels.columns:
            levels[col] = pd.to_datetime(levels[col], utc=True, errors="coerce")

    seg_rows = []
    marker_rows = []

    for row in levels.itertuples(index=False):
        end_idx = int(getattr(row, "end_idx", -1))
        if end_idx < 0 or end_idx >= len(df.index):
            continue
        t_start = df.index[end_idx]
        if t_start < t0:
            continue

        y = getattr(row, "last_extreme", float("nan"))
        if not pd.notna(y):
            y = getattr(row, "price", float("nan"))
        if not pd.notna(y):
            continue

        state = getattr(row, "state", "intact")
        break_idx = getattr(row, "break_idx", float("nan"))
        ft_time = getattr(row, "first_touch_time", pd.NaT)
        update_count = getattr(row, "update_count", 0)

        if state == "broken" and pd.notna(break_idx) and break_idx >= 0:
            t_end = df.index[int(break_idx)]
            if t_end > t1:
                t_end = t1
        else:
            t_end = t1

        seg_rows.append(
            {
                "type": getattr(row, "type", ""),
                "state": state,
                "y": y,
                "t_start": t_start,
                "t_end": t_end,
                "price_at_plot": y,
                "break_idx": break_idx if pd.notna(break_idx) else float("nan"),
                "first_touch_time": ft_time,
                "update_count": int(update_count) if pd.notna(update_count) else 0,
            }
        )

        if pd.notna(ft_time) and t0 <= ft_time <= t1:
            marker_rows.append(
                {
                    "kind": "first_touch",
                    "time": ft_time,
                    "y": y,
                    "level_type": getattr(row, "type", ""),
                    "state": state,
                }
            )

        if state == "broken" and pd.notna(break_idx) and break_idx >= 0:
            break_time = df.index[int(break_idx)]
            if t0 <= break_time <= t1:
                marker_rows.append(
                    {
                        "kind": "break",
                        "time": break_time,
                        "y": y,
                        "level_type": getattr(row, "type", ""),
                        "state": "broken",
                    }
                )

    segments_df = pd.DataFrame(seg_rows, columns=SEG_COLS)
    markers_df = pd.DataFrame(marker_rows, columns=MARKER_COLS)
    return segments_df, markers_df


def plot_levels(
    df: pd.DataFrame,
    segments_df: pd.DataFrame,
    markers_df: Optional[pd.DataFrame],
    cfg: LevelsVizCfg,
    out_png_path: str,
    title: str,
) -> None:
    """Plot levels over price chart and save to PNG."""

    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h), dpi=cfg.dpi)
    ax.plot(df.index, df["close"], color=cfg.colors.get("price", "#6b7280"), lw=1)

    digits = (
        int(round(-math.log10(cfg.tick_size))) if cfg.tick_size > 0 else 5
    )

    for seg in segments_df.itertuples(index=False):
        color_key = f"{seg.type}_{seg.state}" if seg.state in {"broken", "intact"} else f"{seg.type}_intact"
        color = cfg.colors.get(color_key, "black")
        alpha = cfg.alpha_broken if seg.state == "broken" else cfg.alpha_intact
        ax.hlines(seg.y, seg.t_start, seg.t_end, colors=color, lw=cfg.linewidth, alpha=alpha)
        if cfg.show_labels_on_edges:
            ax.text(
                seg.t_end,
                seg.y,
                f"{seg.y:.{digits}f}",
                color=color,
                fontsize=8,
                ha="left",
                va="center",
            )

    if markers_df is not None and not markers_df.empty:
        for m in markers_df.itertuples(index=False):
            marker_color = cfg.colors.get(f"{m.level_type}_{m.state}", "black")
            if m.kind == "break":
                ax.scatter(m.time, m.y, marker="x", color=marker_color)
                if cfg.show_break_vlines:
                    ax.axvline(m.time, color=cfg.colors.get("break_vline", "#94a3b8"), ls="--", lw=1)
            elif m.kind == "first_touch" and cfg.show_first_touch_markers:
                ax.scatter(m.time, m.y, marker="o", color=marker_color)

    ax.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi)
    plt.close(fig)
