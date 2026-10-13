"""Shared helpers for the forecasting package (data loading, calendar features).

Everything here is pure pandas/numpy so it can be imported by both the offline training
scripts and the Django app without pulling in scikit-learn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FORECAST_WEEKS = ["2026-09-28", "2026-10-05", "2026-10-12", "2026-10-19"]
HORIZON = 4


def load_sales() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sales_history.csv", parse_dates=["week_start"])
    return df.sort_values(["sku", "week_start"]).reset_index(drop=True)


def load_lead_times() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "lead_times.csv")


def load_promos() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "promo_calendar.csv", parse_dates=["week_start"])


def diwali_week_starts() -> pd.DatetimeIndex:
    """Monday of the week that contains each Diwali date (festival_calendar.csv)."""
    fest = pd.read_csv(DATA_DIR / "festival_calendar.csv", parse_dates=["date"])
    d = pd.DatetimeIndex(fest.loc[fest.festival.str.lower() == "diwali", "date"])
    return d - pd.to_timedelta(d.dayofweek, unit="D")


def weeks_to_diwali(week_start: pd.Series) -> pd.Series:
    """Signed whole weeks from each week to the *nearest* Diwali week.

    Negative = the week is before Diwali (-3 means Diwali week is 3 weeks away), 0 = the
    Diwali week itself, positive = after. The festive demand lift follows this number, not the
    calendar month, because Diwali moves by up to ~3 weeks between years.
    """
    fw = diwali_week_starts()
    wk = pd.DatetimeIndex(week_start)
    diffs = np.stack([((wk - f).days // 7).to_numpy() for f in fw])        # (n_fest, n)
    idx = np.abs(diffs).argmin(axis=0)
    return pd.Series(diffs[idx, np.arange(len(wk))], index=getattr(week_start, "index", None))


def forecast_frame() -> pd.DataFrame:
    """The 160 (sku, week, promo) rows we must predict. Promotions are known in advance."""
    p = load_promos().copy()
    p["week_start"] = pd.to_datetime(p["week_start"])
    return p.sort_values(["sku", "week_start"]).reset_index(drop=True)
