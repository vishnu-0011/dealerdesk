"""Rolling-origin backtest: every fold trains only on data strictly before its origin."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import HORIZON, load_sales


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    return {"MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "WAPE": np.abs(err).sum() / max(y.sum(), 1e-9), "bias": err.mean()}


def run_fold(sales: pd.DataFrame, origin: pd.Timestamp, models: dict) -> pd.DataFrame:
    # LEAKAGE GUARD: the training set is cut strictly before the origin week. Nothing at or
    # after `origin` (not even its promo flags or units) is visible when the models are fit.
    train = sales[sales.week_start < origin]
    end = origin + pd.Timedelta(weeks=HORIZON)
    future = sales[(sales.week_start >= origin) & (sales.week_start < end)].copy()
    out = future[["week_start", "sku", "units_sold"]].copy()
    out["origin"] = origin
    out["h"] = ((future.week_start - origin).dt.days // 7 + 1).to_numpy()
    fut_in = future[["week_start", "sku", "promo_flag"]]          # promo flag is known in advance
    for name, fn in models.items():
        out[name] = fn(train, fut_in)
    return out


def rolling_backtest(models: dict, origins) -> pd.DataFrame:
    sales = load_sales()
    return pd.concat([run_fold(sales, pd.Timestamp(o), models) for o in origins], ignore_index=True)


def summarise(bt: pd.DataFrame, names) -> pd.DataFrame:
    rows = {n: metrics(bt.units_sold, bt[n]) for n in names}
    return pd.DataFrame(rows).T.round(4)
