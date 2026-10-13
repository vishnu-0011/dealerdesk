from django.core.management import call_command
from django.test import TestCase
from rest_framework.test import APIClient

from .models import Dealer, Inventory, Product


class CatalogAPITests(TestCase):
    @classmethod
    def setUpTestData(cls):
        call_command("seed_data", verbosity=0)

    def setUp(self):
        self.c = APIClient()

    def test_seed_is_idempotent_and_keeps_live_stock(self):
        self.c.put("/api/inventory/BRK-1044/", {"quantity": 7}, format="json")
        call_command("seed_data", verbosity=0)
        self.assertEqual(Product.objects.count(), 600)
        self.assertEqual(Dealer.objects.count(), 25)
        self.assertEqual(Inventory.objects.get(product__sku="BRK-1044").quantity, 7)

    def test_product_list_pagination_and_filters(self):
        r = self.c.get("/api/products/?page_size=5").json()
        self.assertEqual((r["count"], len(r["results"])), (600, 5))
        self.assertIsInstance(r["results"][0]["price"], str)
        self.assertRegex(r["results"][0]["price"], r"^\d+\.\d{2}$")
        duke = self.c.get("/api/products/?vehicle=duke%20390&page_size=100").json()
        self.assertTrue(duke["results"] and all("Duke 390" in p["vehicle_fitment"] for p in duke["results"]))
        self.assertEqual(self.c.get("/api/products/?search=BRK-1044").json()["count"], 1)

    def test_product_create_validation_and_duplicates(self):
        ok = self.c.post("/api/products/", {"sku": "TST-0001", "name": "Test", "price": "10.50"}, format="json")
        self.assertEqual((ok.status_code, ok.json()["stock"], ok.json()["price"]), (201, 0, "10.50"))
        dup = self.c.post("/api/products/", {"sku": "TST-0001", "name": "x", "price": 5}, format="json")
        self.assertEqual((dup.status_code, dup.json()["error"]["code"]), (409, "DUPLICATE"))
        for bad in ({"sku": "TST-0002", "name": "x", "price": 0}, {"sku": "TST-0002", "name": "x", "price": 1.234},
                    {"sku": "TST-0002", "name": "x", "price": -4}, {"sku": "TST-0002", "name": "", "price": 4},
                    {"name": "x", "price": 4}, {"sku": "TST-0002", "name": "x", "price": True}):
            r = self.c.post("/api/products/", bad, format="json")
            self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "VALIDATION_ERROR"), bad)

    def test_patch_product(self):
        r = self.c.patch("/api/products/BRK-1044/", {"price": "99.99"}, format="json")
        self.assertEqual((r.status_code, r.json()["price"]), (200, "99.99"))
        self.assertEqual(self.c.patch("/api/products/BRK-1044/", {"sku": "OTHER-1"}, format="json").status_code, 400)
        self.assertEqual(self.c.patch("/api/products/NOPE-0000/", {"price": 5}, format="json").status_code, 404)

    def test_inventory_put(self):
        r = self.c.put("/api/inventory/BRK-1044/", {"quantity": 25, "reason": "cycle count"}, format="json")
        self.assertEqual((r.status_code, r.json()), (200, {"sku": "BRK-1044", "quantity": 25}))
        self.assertEqual(self.c.get("/api/products/BRK-1044/").json()["stock"], 25)
        for bad in (-1, 2.5, "7", None, True):
            self.assertEqual(self.c.put("/api/inventory/BRK-1044/", {"quantity": bad}, format="json").status_code, 400, bad)
        self.assertEqual(self.c.put("/api/inventory/NOPE-0000/", {"quantity": 1}, format="json").status_code, 404)

    def test_dealers(self):
        r = self.c.post("/api/dealers/", {"dealer_code": "D-T1", "name": "T", "credit_limit": "1000.00"}, format="json")
        self.assertEqual((r.status_code, r.json()["credit_limit"]), (201, "1000.00"))
        self.assertEqual(self.c.post("/api/dealers/", {"dealer_code": "D-T1", "name": "T"}, format="json").status_code, 409)
        self.assertEqual(self.c.get(f"/api/dealers/{r.json()['id']}/").status_code, 200)
        self.assertEqual(self.c.get("/api/dealers/999999/").json()["error"]["code"], "NOT_FOUND")
        self.assertEqual(self.c.get("/api/dealers/?page_size=10").json()["count"], 26)

    def test_unknown_route_and_bad_json_use_error_envelope(self):
        self.assertEqual(self.c.get("/api/nope/").json()["error"]["code"], "NOT_FOUND")
        r = self.c.post("/api/products/", "{not json", content_type="application/json")
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "VALIDATION_ERROR"))
