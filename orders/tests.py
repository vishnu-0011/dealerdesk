import threading
from decimal import Decimal

from django.core.management import call_command
from django.db import connection
from django.test import TestCase, TransactionTestCase
from rest_framework.test import APIClient

from catalog.models import Dealer, Product
from orders.services import price_line, to_paisa


def make_catalog():
    call_command("seed_data", verbosity=0)
    return Dealer.objects.first()


class MoneyRules(TestCase):
    def test_round_half_up_not_half_even(self):
        self.assertEqual(to_paisa(Decimal("1.845")), Decimal("1.85"))       # half-even would give 1.84
        self.assertEqual(to_paisa(Decimal("0.125")), Decimal("0.13"))
        line = price_line(Decimal("10.25"), 1)
        self.assertEqual((line["line_tax"], line["line_total"]), (Decimal("1.85"), Decimal("12.10")))

    def test_gst_is_per_line(self):
        a, b = price_line(Decimal("0.25"), 1), price_line(Decimal("0.25"), 1)
        whole = price_line(Decimal("0.25"), 2)
        self.assertEqual(a["line_tax"] + b["line_tax"], Decimal("0.10"))     # 0.045 -> 0.05 each
        self.assertEqual(whole["line_tax"], Decimal("0.09"))                 # per-line rule on 0.50


class OrderAPITests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.dealer = make_catalog()

    def setUp(self):
        self.c = APIClient()

    def post(self, items, dealer=None):
        return self.c.post("/api/orders/", {"dealer_id": dealer or self.dealer.id, "items": items}, format="json")

    def test_create_and_read(self):
        r = self.post([{"sku": "BRK-1044", "quantity": 3}, {"sku": "BRK-1032", "quantity": 1}])
        self.assertEqual(r.status_code, 201)
        o = r.json()
        self.assertRegex(o["order_number"], r"^ORD-\d{8}-0001$")
        self.assertEqual(o["status"], "DRAFT")
        p = Product.objects.get(sku="BRK-1044")
        self.assertEqual(o["items"][0]["unit_price"], f"{p.price:.2f}")
        total = sum(Decimal(i["line_total"]) for i in o["items"])
        self.assertEqual(Decimal(o["grand_total"]), total)
        self.assertEqual(Decimal(o["subtotal"]) + Decimal(o["tax_total"]), Decimal(o["grand_total"]))
        self.assertEqual(self.c.get(f"/api/orders/{o['order_number']}/").json(), o)
        self.assertEqual(self.post([{"sku": "BRK-1044", "quantity": 1}]).json()["order_number"][-4:], "0002")

    def test_validation(self):
        bad = [[], None, [{"sku": "BRK-1044", "quantity": 0}], [{"sku": "BRK-1044", "quantity": 1.5}],
               [{"sku": "BRK-1044", "quantity": "2"}], [{"sku": "NOPE-9999", "quantity": 1}],
               [{"sku": "BRK-1044", "quantity": 1}, {"sku": "BRK-1044", "quantity": 2}], [{"quantity": 1}]]
        for items in bad:
            r = self.post(items)
            self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "VALIDATION_ERROR"), items)
        self.assertEqual(self.post([{"sku": "BRK-1044", "quantity": 1}], dealer=999999).status_code, 400)

    def test_draft_does_not_touch_stock_and_follows_price(self):
        before = Product.objects.select_related("inventory").get(sku="BRK-1044")
        o = self.post([{"sku": "BRK-1044", "quantity": 2}]).json()
        self.assertEqual(Product.objects.select_related("inventory").get(sku="BRK-1044").stock, before.stock)
        self.c.patch("/api/products/BRK-1044/", {"price": "100.00"}, format="json")
        again = self.c.get(f"/api/orders/{o['order_number']}/").json()
        self.assertEqual((again["items"][0]["unit_price"], again["subtotal"], again["tax_total"]), ("100.00", "200.00", "36.00"))

    def test_list_filters_and_pagination(self):
        for _ in range(3):
            self.post([{"sku": "BRK-1044", "quantity": 1}])
        r = self.c.get(f"/api/orders/?dealer_id={self.dealer.id}&status=DRAFT&page_size=2").json()
        self.assertEqual((r["count"], len(r["results"])), (3, 2))
        self.assertEqual(self.c.get("/api/orders/?status=BOGUS").status_code, 400)
        self.assertEqual(self.c.get("/api/orders/NOPE/").status_code, 404)


class OrderNumberConcurrency(TransactionTestCase):
    def test_parallel_creates_get_unique_gapless_numbers(self):
        dealer = make_catalog()
        numbers, errors = [], []

        def worker():
            try:
                c = APIClient()
                r = c.post("/api/orders/", {"dealer_id": dealer.id, "items": [{"sku": "BRK-1044", "quantity": 1}]}, format="json")
                numbers.append(r.json()["order_number"])
            except Exception as e:  # pragma: no cover
                errors.append(e)
            finally:
                connection.close()

        threads = [threading.Thread(target=worker) for _ in range(12)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertFalse(errors)
        self.assertEqual(len(set(numbers)), 12)
        self.assertEqual(sorted(int(n[-4:]) for n in numbers), list(range(1, 13)))
