"""Serve ml/forecast.csv and compute reorder suggestions (contract section 10).

Runtime has no pandas/scikit-learn dependency: the model is trained offline (ml/train_forecast.py)
and this module only reads its committed output plus data/sales_history.csv and lead_times.csv.
"""
import csv
import math
import statistics
import threading
from collections import defaultdict
from decimal import Decimal

from django.conf import settings

from catalog.models import Inventory

SAFETY_Z = 1.65
SIGMA_WEEKS = 12
_lock = threading.Lock()
_cache: dict = {}


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _cached(path, loader):
    """Re-read a file only when its modification time changes."""
    mtime = path.stat().st_mtime
    with _lock:
        hit = _cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        value = loader(_read(path))
        _cache[path] = (mtime, value)
        return value


def forecasts() -> dict[str, list[tuple[str, Decimal]]]:
    """sku -> [(week_start, forecast_units)] in week order."""
    def load(rows):
        out = defaultdict(list)
        for r in rows:
            out[r["sku"]].append((r["week_start"], Decimal(r["forecast_units"])))
        return {k: sorted(v) for k, v in out.items()}
    return _cached(settings.FORECAST_CSV, load)


def lead_times() -> dict[str, int]:
    return _cached(settings.DATA_DIR / "lead_times.csv", lambda rows: {r["sku"]: int(r["lead_time_weeks"]) for r in rows})


def recent_sales() -> dict[str, list[int]]:
    """sku -> units_sold for its most recent SIGMA_WEEKS weeks (oldest first)."""
    def load(rows):
        by_sku = defaultdict(list)
        for r in rows:
            by_sku[r["sku"]].append((r["week_start"], int(r["units_sold"])))
        return {k: [u for _, u in sorted(v)[-SIGMA_WEEKS:]] for k, v in by_sku.items()}
    return _cached(settings.DATA_DIR / "sales_history.csv", load)


def forecast_for(sku: str):
    """Return the 4-week forecast list for a forecast SKU, or None when it is not one."""
    weeks = forecasts().get(sku)
    if not weeks:
        return None
    return [{"week_start": w, "forecast_units": float(u)} for w, u in weeks]


def reorder_entry(sku: str, current_stock: int) -> dict:
    L = lead_times()[sku]
    weeks = forecasts()[sku]
    lead_time_demand = sum((u for _, u in weeks[:L]), Decimal("0"))          # exact decimal sum
    history = recent_sales().get(sku, [])
    sigma = statistics.stdev(history) if len(history) >= 2 else 0.0           # sample std (n-1)
    safety_stock = math.ceil(SAFETY_Z * sigma * math.sqrt(L))
    reorder_point = lead_time_demand + safety_stock
    reorder_qty = max(0, math.ceil(reorder_point - current_stock))            # Decimal - int is exact
    return {"sku": sku, "current_stock": current_stock, "lead_time_weeks": L,
            "lead_time_demand": float(lead_time_demand), "safety_stock": safety_stock,
            "reorder_point": float(reorder_point), "reorder_qty": reorder_qty}


def reorder_suggestions() -> list[dict]:
    skus = list(lead_times())
    live = dict(Inventory.objects.filter(product__sku__in=skus).values_list("product__sku", "quantity"))  # live stock
    return [reorder_entry(sku, live.get(sku, 0)) for sku in skus]
