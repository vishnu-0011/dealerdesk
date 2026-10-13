"""Train on all history, write ml/forecast.csv, and write the backtest report.

    python -m ml.train_forecast

Reproducible: no randomness (Poisson GLM, deterministic L-BFGS). Needs scikit-learn and pandas
(see ml/requirements.txt); the Django app itself does NOT need them, it only reads forecast.csv.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from .backtest import metrics, rolling_backtest
from .common import FORECAST_WEEKS, ROOT, forecast_frame, load_lead_times, load_sales
from .models import BASELINES, ForecastModel, GradientBoosting, PoissonGLM

warnings.filterwarnings("ignore")
OUT = ROOT / "ml"


def write_forecast() -> pd.DataFrame:
    sales = load_sales()
    future = forecast_frame()
    first_forecast_week = pd.Timestamp(FORECAST_WEEKS[0])
    # LEAKAGE GUARD (production): the model is fitted only on weeks strictly before the first
    # forecast week; the forecast rows carry nothing but (sku, week, promo flag).
    train = sales[sales.week_start < first_forecast_week]
    assert train.week_start.max() < first_forecast_week
    pred = ForecastModel()(train, future[["week_start", "sku", "promo_flag"]])
    out = pd.DataFrame({"sku": future.sku, "week_start": future.week_start.dt.strftime("%Y-%m-%d"),
                        "forecast_units": np.round(np.clip(pred, 0, None), 2)})
    skus = set(load_lead_times().sku)
    assert set(out.sku) == skus and len(out) == len(skus) * 4 and (out.forecast_units >= 0).all()
    out.to_csv(OUT / "forecast.csv", index=False)
    return out


def backtest_report() -> dict:
    models = dict(BASELINES)
    models["gradient_boosting"] = GradientBoosting()
    models["glm_plain"] = PoissonGLM(alpha=0.1, promo_sku_scale=0.8)
    models["forecast_model"] = ForecastModel()
    names = list(models)
    rolling = rolling_backtest(models, pd.date_range("2025-09-29", "2026-08-17", freq="14D"))
    # the fold that mirrors the real task: forecast weeks 5,4,3,2 before Diwali, one festive
    # season of history (same-week-last-year is unavailable there and falls back to last value).
    mirror = rolling_backtest(models, [pd.Timestamp("2025-09-15")])
    rep = {}
    for label, bt in (("rolling_24_folds", rolling), ("mirror_fold_2025-09-15", mirror)):
        rep[label] = {n: {k: round(v, 4) for k, v in metrics(bt.units_sold, bt[n]).items()} for n in names}
    rep["wape_by_horizon"] = {int(h): {n: round(metrics(g.units_sold, g[n])["WAPE"], 4) for n in
                                      ("recent_avg_4w", "forecast_model")} for h, g in rolling.groupby("h")}
    return rep


if __name__ == "__main__":
    f = write_forecast()
    print(f"wrote ml/forecast.csv: {len(f)} rows")
    rep = backtest_report()
    (OUT / "backtest_report.json").write_text(json.dumps(rep, indent=2))
    for k in ("rolling_24_folds", "mirror_fold_2025-09-15"):
        print("\n" + k)
        print(pd.DataFrame(rep[k]).T.to_string())
    print("\nWAPE by horizon:", rep["wape_by_horizon"])
