import csv
import math
import statistics

from django.conf import settings
from django.core.management import call_command
from django.test import TestCase
from rest_framework.test import APIClient

from forecasting import service


class ForecastAPITests(TestCase):
    @classmethod
    def setUpTestData(cls):
        call_command("seed_data", verbosity=0)

    def setUp(self):
        self.c = APIClient()

    def test_forecast_shape(self):
        r = self.c.get("/api/forecast/FIL-1016/").json()
        self.assertEqual(r["sku"], "FIL-1016")
        self.assertEqual([f["week_start"] for f in r["forecast"]], ["2026-09-28", "2026-10-05", "2026-10-12", "2026-10-19"])
        self.assertTrue(all(isinstance(f["forecast_units"], float) and f["forecast_units"] >= 0 for f in r["forecast"]))
        self.assertEqual(self.c.get("/api/forecast/BRK-1001/").status_code, 404)    # not a forecast SKU

    def test_forecast_csv_covers_every_sku_and_week(self):
        rows = list(csv.DictReader(open(settings.FORECAST_CSV)))
        lead = list(csv.DictReader(open(settings.DATA_DIR / "lead_times.csv")))
        self.assertEqual(len(rows), 160)
        self.assertEqual({r["sku"] for r in rows}, {r["sku"] for r in lead})

    def test_reorder_matches_independent_calculation(self):
        sales = {}
        for r in csv.DictReader(open(settings.DATA_DIR / "sales_history.csv")):
            sales.setdefault(r["sku"], []).append((r["week_start"], int(r["units_sold"])))
        fc = {}
        for r in csv.DictReader(open(settings.FORECAST_CSV)):
            fc.setdefault(r["sku"], []).append((r["week_start"], float(r["forecast_units"])))
        self.c.put("/api/inventory/FIL-1016/", {"quantity": 40}, format="json")   # live stock must be used
        results = {x["sku"]: x for x in self.c.get("/api/reorder-suggestions/").json()["results"]}
        self.assertEqual(len(results), 40)
        for lt in csv.DictReader(open(settings.DATA_DIR / "lead_times.csv")):
            sku, L = lt["sku"], int(lt["lead_time_weeks"])
            last12 = [u for _, u in sorted(sales[sku])[-12:]]
            ltd = sum(u for _, u in sorted(fc[sku])[:L])
            ss = math.ceil(1.65 * statistics.stdev(last12) * math.sqrt(L))
            got = results[sku]
            self.assertEqual((got["lead_time_weeks"], got["safety_stock"]), (L, ss), sku)
            self.assertAlmostEqual(got["lead_time_demand"], ltd, places=2)
            self.assertAlmostEqual(got["reorder_point"], ltd + ss, places=2)
            self.assertEqual(got["reorder_qty"], max(0, math.ceil(round(ltd + ss, 2) - got["current_stock"])), sku)
        self.assertEqual(results["FIL-1016"]["current_stock"], 40)

    def test_reorder_qty_never_negative_and_zero_sigma(self):
        e = service.reorder_entry("FIL-1016", 10 ** 6)
        self.assertEqual(e["reorder_qty"], 0)
