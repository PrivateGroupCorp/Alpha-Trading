from __future__ import annotations

import pandas as pd


def _tf_to_timedelta(tf: str) -> pd.Timedelta:
    unit = tf[0]
    num = int(tf[1:])
    mapping = {"S": "seconds", "M": "minutes", "H": "hours", "D": "days"}
    return pd.Timedelta(**{mapping[unit]: num})


def quality_report(df: pd.DataFrame, tf: str) -> dict:
    expected_step = _tf_to_timedelta(tf)
    diff = df.index.to_series().diff().dropna()
    gaps = diff[diff >= 2 * expected_step]
    report = {
        "span": [
            df.index.min().isoformat() if not df.empty else None,
            df.index.max().isoformat() if not df.empty else None,
        ],
        "n_rows": int(len(df)),
        "expected_step_sec": expected_step.total_seconds(),
        "gaps": {
            "count": int(len(gaps)),
            "percent": float(len(gaps) / len(df)) if len(df) else 0.0,
        },
        "duplicates_removed": int(df.index.duplicated().sum()),
        "nan_rows": int(df.isna().any(axis=1).sum()),
    }
    return report
