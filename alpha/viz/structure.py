"""Visualization utilities for market structure (swings and BOS/CHoCH events)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class StructureVizCfg:
    last_n_bars: int = 500
    zigzag_linewidth: float = 1.8
    price_linewidth: float = 1.0
    swing_marker_size: int = 28
    event_marker_size: int = 60
    alpha_zigzag: float = 0.9
    alpha_price: float = 0.8
    colors: dict = field(
        default_factory=lambda: {
            "price": "#6b7280",
            "swing_peak": "#ef4444",
            "swing_trough": "#0ea5e9",
            "bos_up": "#22c55e",
            "bos_down": "#ef4444",
            "choch_up": "#a3e635",
            "choch_down": "#fb7185",
            "invalid_event": "#9ca3af",
            "zigzag": "#6b7280",
        }
    )
    grade_palette: dict = field(
        default_factory=lambda: {
            "A": "#22c55e",
            "B": "#f59e0b",
            "C": "#ef4444",
        }
    )
    show_only_valid: bool = True
    annotate_labels: bool = True
    show_quality_grade: bool = True
    show_ft_window: bool = False
    dpi: int = 150
    fig_w: float = 12.0
    fig_h: float = 6.0


SEGMENT_COLS = [
    "seg_kind",
    "x_start",
    "x_end",
    "y_start",
    "y_end",
    "meta",
]

MARKER_COLS = [
    "mark_kind",
    "time",
    "y",
    "grade",
    "is_valid",
    "event_id",
    "swing_id",
    "ft_bars",
]


def build_structure_segments_and_markers(
    df: pd.DataFrame,
    swings_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_last_n: Optional[int] = 500,
    show_only_valid: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build segment and marker frames for structure visualization."""

    if window_last_n and window_last_n > 0:
        tail_df = df.tail(window_last_n)
    else:
        tail_df = df
    if tail_df.empty:
        return (
            pd.DataFrame(columns=SEGMENT_COLS),
            pd.DataFrame(columns=MARKER_COLS),
        )

    t0, t1 = tail_df.index[0], tail_df.index[-1]

    swings = swings_df.copy()
    if "time" in swings.columns:
        swings["time"] = pd.to_datetime(swings["time"], utc=True, errors="coerce")
    swings = swings.sort_values("time")

    events = events_df.copy()
    if "time" in events.columns:
        events["time"] = pd.to_datetime(events["time"], utc=True, errors="coerce")

    seg_rows: list[dict] = []
    marker_rows: list[dict] = []

    # Price segments
    idx = tail_df.index
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

    # ZigZag segments between swings
    for i in range(len(swings) - 1):
        row0 = swings.iloc[i]
        row1 = swings.iloc[i + 1]
        t_start = row0.get("time")
        t_end = row1.get("time")
        if pd.isna(t_start) or pd.isna(t_end):
            continue
        if t_end < t0 or t_start > t1:
            continue
        seg_rows.append(
            {
                "seg_kind": "zigzag",
                "x_start": max(t_start, t0),
                "x_end": min(t_end, t1),
                "y_start": float(row0.get("price", float("nan"))),
                "y_end": float(row1.get("price", float("nan"))),
                "meta": json.dumps(
                    {
                        "from_swing_id": int(row0.get("swing_id", -1)),
                        "to_swing_id": int(row1.get("swing_id", -1)),
                    }
                ),
            }
        )

    # Swing markers
    for row in swings.itertuples(index=False):
        t = getattr(row, "time", pd.NaT)
        if pd.isna(t) or t < t0 or t > t1:
            continue
        kind = "swing_peak" if getattr(row, "type", "") == "peak" else "swing_trough"
        marker_rows.append(
            {
                "mark_kind": kind,
                "time": t,
                "y": float(getattr(row, "price", float("nan"))),
                "grade": "",
                "is_valid": pd.NA,
                "event_id": pd.NA,
                "swing_id": int(getattr(row, "swing_id", -1)),
                "ft_bars": pd.NA,
            }
        )

    # Event markers
    events_win = events[(events["time"] >= t0) & (events["time"] <= t1)]
    has_quality = {"is_valid", "quality_grade"}.issubset(events_win.columns)
    if has_quality and show_only_valid:
        events_win = events_win[events_win["is_valid"] == True]
    for row in events_win.itertuples(index=False):
        kind = f"{row.event.lower()}_{row.direction}"
        grade = getattr(row, "quality_grade", "") if has_quality else ""
        is_valid = getattr(row, "is_valid", pd.NA) if has_quality else getattr(row, "is_valid", pd.NA)
        marker_rows.append(
            {
                "mark_kind": kind,
                "time": getattr(row, "time"),
                "y": float(getattr(row, "ref_price", float("nan"))),
                "grade": grade,
                "is_valid": bool(is_valid) if pd.notna(is_valid) else pd.NA,
                "event_id": int(getattr(row, "event_id", -1)),
                "swing_id": pd.NA,
                "ft_bars": getattr(row, "ft_bars", pd.NA),
            }
        )

    segments_df = pd.DataFrame(seg_rows, columns=SEGMENT_COLS)
    markers_df = pd.DataFrame(marker_rows, columns=MARKER_COLS)
    return segments_df, markers_df


def plot_structure(
    df: pd.DataFrame,
    segments_df: pd.DataFrame,
    markers_df: pd.DataFrame,
    cfg: StructureVizCfg,
    out_png_path: str,
    title: str,
) -> None:
    """Plot price, swings, and events, and save to PNG."""

    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h), dpi=cfg.dpi)
    # Plot price line
    ax.plot(
        df.index,
        df["close"],
        color=cfg.colors.get("price", "#6b7280"),
        lw=cfg.price_linewidth,
        alpha=cfg.alpha_price,
    )

    # Plot zigzag segments
    zigzag_seg = segments_df[segments_df["seg_kind"] == "zigzag"]
    for seg in zigzag_seg.itertuples(index=False):
        ax.plot(
            [seg.x_start, seg.x_end],
            [seg.y_start, seg.y_end],
            color=cfg.colors.get("zigzag", cfg.colors.get("price", "#6b7280")),
            lw=cfg.zigzag_linewidth,
            alpha=cfg.alpha_zigzag,
        )

    marker_style = {
        "swing_peak": ("^", "none"),
        "swing_trough": ("v", "none"),
        "bos_up": ("^", "filled"),
        "bos_down": ("v", "filled"),
        "choch_up": ("D", "none"),
        "choch_down": ("D", "filled"),
    }

    for m in markers_df.itertuples(index=False):
        kind = m.mark_kind
        base_color = cfg.colors.get(kind, cfg.colors.get("invalid_event", "#9ca3af"))
        if getattr(m, "is_valid", True) is False:
            color = cfg.colors.get("invalid_event", "#9ca3af")
        else:
            color = base_color
            grade = getattr(m, "grade", "")
            if grade and cfg.grade_palette:
                color = cfg.grade_palette.get(grade, color)
        marker, fill = marker_style.get(kind, ("o", "filled"))
        size = cfg.swing_marker_size if kind.startswith("swing") else cfg.event_marker_size
        facecolor = color if fill == "filled" else "none"
        ax.scatter(
            m.time,
            m.y,
            s=size,
            marker=marker,
            edgecolors=color,
            facecolors=facecolor,
            zorder=3,
        )

        label = ""
        if cfg.annotate_labels:
            if kind.startswith("swing"):
                prefix = "P" if kind == "swing_peak" else "T"
                if pd.notna(m.swing_id):
                    label = f"{prefix}{int(m.swing_id)}"
            else:
                base = "BOS" if kind.startswith("bos") else "CHoCH"
                arrow = "↑" if kind.endswith("up") else "↓"
                label = f"{base}{arrow}"
                if cfg.show_quality_grade and getattr(m, "grade", ""):
                    label += f" {m.grade}"
        if label:
            ax.text(m.time, m.y, label, color=color, fontsize=8, ha="left", va="bottom")

        if (
            cfg.show_ft_window
            and not pd.isna(getattr(m, "ft_bars", pd.NA))
            and not kind.startswith("swing")
        ):
            try:
                idx0 = df.index.get_loc(m.time)
                idx1 = min(len(df) - 1, idx0 + int(m.ft_bars))
                ax.axvspan(
                    df.index[idx0],
                    df.index[idx1],
                    color=color,
                    alpha=0.1,
                )
            except KeyError:
                pass

    ax.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi)
    plt.close(fig)
