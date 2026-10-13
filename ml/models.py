"""Forecasting models and the three baselines the grader compares us against.

Every model has the same signature:  fn(train, future) -> np.ndarray of predictions aligned
with ``future``. ``train`` has columns [week_start, sku, units_sold, promo_flag]; ``future`` has
[week_start, sku, promo_flag] (promotions are known in advance). A model may only look at
``train`` - the backtest harness (backtest.py) is what guarantees ``train`` ends before the
forecast origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

from .common import weeks_to_diwali

# --------------------------------------------------------------------------- baselines
def _last_value(train: pd.DataFrame) -> pd.Series:
    return train.sort_values("week_start").groupby("sku").units_sold.last()


def baseline_last_value(train, future):
    return future.sku.map(_last_value(train)).fillna(0).to_numpy(float)


def baseline_recent_avg(train, future, n: int = 4):
    avg = train.sort_values("week_start").groupby("sku").units_sold.apply(lambda s: s.tail(n).mean())
    return future.sku.map(avg).fillna(0).to_numpy(float)


def baseline_same_week_last_year(train, future):
    """Value 52 weeks earlier; falls back to the last observed value when there is none."""
    key = train.set_index(["sku", "week_start"]).units_sold
    prev = future.week_start - pd.Timedelta(weeks=52)
    lookup = key.to_dict()
    got = np.array([lookup.get((k, w), np.nan) for k, w in zip(future.sku, prev)], dtype=float)
    fallback = future.sku.map(_last_value(train)).fillna(0).to_numpy(float)
    return np.where(np.isnan(got), fallback, got)


BASELINES = {
    "last_value": baseline_last_value,
    "recent_avg_4w": baseline_recent_avg,
    "same_week_last_year": baseline_same_week_last_year,
}

# --------------------------------------------------------------------------- GLM
FESTIVE_REL = list(range(-4, 4))           # weeks-to-Diwali with their own effect; others = baseline


class PoissonGLM:
    """Multiplicative demand model:  E[units] = exp(sku + trend + festive[rel] + promo ...).

    * sku            one level per SKU (weakly regularised so a 6-week-old SKU still gets its own)
    * trend          one global growth rate per year
    * festive[rel]   one pooled multiplier per week-relative-to-Diwali (-4..+3), shared by all SKUs
    * promo          a pooled promo multiplier plus a heavily-shrunk per-SKU deviation
    * (optional) annual Fourier terms, category x festive terms
    Fitted with a Poisson log-link: counts, never negative, noise grows with the mean.
    """

    def __init__(self, alpha=1.0, sku_scale=10.0, festive_scale=3.0, promo_scale=3.0,
                 promo_sku_scale=0.8, fourier_k=0, cat_festive_scale=0.0, trend=True,
                 recency_weight=0.0):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        cols = []
        sku_idx = df.sku.map(self.sku_index_).to_numpy()
        S = np.zeros((n, len(self.skus_)))
        S[np.arange(n), sku_idx] = self.sku_scale
        cols.append(S)
        if self.trend:
            t = ((df.week_start - self.t0_).dt.days / 364.0).to_numpy()[:, None]
            cols.append(t)
        rel = weeks_to_diwali(df.week_start).to_numpy()
        F = np.stack([(rel == r).astype(float) for r in FESTIVE_REL], axis=1) * self.festive_scale
        cols.append(F)
        promo = df.promo_flag.to_numpy(float)[:, None]
        cols.append(promo * self.promo_scale)
        P = np.zeros((n, len(self.skus_)))
        P[np.arange(n), sku_idx] = promo[:, 0] * self.promo_sku_scale
        cols.append(P)
        if self.fourier_k:
            doy = df.week_start.dt.dayofyear.to_numpy() / 365.25 * 2 * np.pi
            for k in range(1, self.fourier_k + 1):
                cols += [np.sin(k * doy)[:, None], np.cos(k * doy)[:, None]]
        if self.cat_festive_scale:
            cats = sorted({s[:3] for s in self.skus_})
            cat = df.sku.str[:3].to_numpy()
            for c in cats:
                for r in range(-4, 1):
                    cols.append((((cat == c) & (rel == r)).astype(float) * self.cat_festive_scale)[:, None])
        return np.hstack(cols)

    def __call__(self, train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        self.skus_ = sorted(set(train.sku) | set(future.sku))
        self.sku_index_ = {s: i for i, s in enumerate(self.skus_)}
        self.t0_ = train.week_start.min()
        X = self._design(train)
        w = None
        if self.recency_weight:
            age = (train.week_start.max() - train.week_start).dt.days / 364.0
            w = np.exp(-self.recency_weight * age.to_numpy())
        m = PoissonRegressor(alpha=self.alpha, fit_intercept=False, max_iter=2000, tol=1e-8)
        m.fit(X, train.units_sold.to_numpy(float), sample_weight=w)
        self.model_ = m
        return m.predict(self._design(future))


# --------------------------------------------------------------------------- GBM challenger
class GradientBoosting:
    """Challenger: Poisson gradient boosting on calendar/promo features + SKU level.

    No lag features: with a 4-week horizon a direct multi-horizon lag model needs one training
    set per horizon, and the data (2 years x 40 SKUs) is small. It is here to show that a flexible
    learner does not beat the structured GLM, see DECISIONS.md.
    """

    def __call__(self, train, future):
        base = train[(train.promo_flag == 0)].groupby("sku").units_sold.mean()
        recent = train.sort_values("week_start").groupby("sku").units_sold.apply(lambda s: s.tail(12).mean())

        def feats(df):
            rel = weeks_to_diwali(df.week_start).clip(-8, 6).to_numpy()
            t = ((df.week_start - train.week_start.min()).dt.days / 364.0).to_numpy()
            lvl = np.log1p(df.sku.map(base).fillna(df.sku.map(recent)).fillna(1).to_numpy())
            rec = np.log1p(df.sku.map(recent).fillna(1).to_numpy())
            return np.column_stack([lvl, rec, rel, df.promo_flag.to_numpy(), t])

        m = HistGradientBoostingRegressor(loss="poisson", max_depth=4, learning_rate=0.05,
                                          max_iter=300, min_samples_leaf=20, random_state=0)
        m.fit(feats(train), train.units_sold.to_numpy(float))
        return m.predict(feats(future))


# --------------------------------------------------------------------------- production model
COLD_START_WEEKS = 26     # SKUs with less history than this get a recent-level correction
COLD_START_TAIL = 3       # ... based on the last 3 observed weeks


class ForecastModel(PoissonGLM):
    """The model we ship: Poisson GLM + a recent-level correction for young SKUs.

    A SKU with only a few weeks of history gets a shrunken SKU effect from the regularised GLM,
    so we rescale its predictions by (actual / fitted) over its most recent weeks. Established
    SKUs are left to the GLM: their level is pinned down by two years of data.
    """

    def __init__(self):
        super().__init__(alpha=0.1, promo_sku_scale=0.8)

    def __call__(self, train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        pred = super().__call__(train, future)
        fitted = train.assign(fit=self.model_.predict(self._design(train)))
        counts = train.groupby("sku").size()
        for sku in counts[counts < COLD_START_WEEKS].index:
            tail = fitted[fitted.sku == sku].sort_values("week_start").tail(COLD_START_TAIL)
            ratio = float(np.clip(tail.units_sold.sum() / max(tail.fit.sum(), 1e-9), 0.5, 2.0))
            pred[(future.sku == sku).to_numpy()] *= ratio
        return pred
