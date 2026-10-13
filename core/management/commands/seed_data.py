"""Load data/catalogue.csv and data/dealers.csv. Safe to run any number of times.

* Products and dealers are upserted by their natural keys (sku / dealer_code): no duplicates.
* Inventory rows are created only for NEW products. A re-run never resets live stock.
* The CSVs are treated as untrusted: values are stripped, bad rows are skipped and reported.
"""
import csv
from decimal import Decimal, InvalidOperation

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from catalog.models import Dealer, Inventory, Product


def _rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        yield from csv.DictReader(fh)


def _money(raw, allow_zero=False):
    try:
        value = Decimal(str(raw).strip())
    except InvalidOperation:
        return None
    if value < 0 or (value == 0 and not allow_zero) or value != value.quantize(Decimal("0.01")):
        return None
    return value


class Command(BaseCommand):
    help = "Load the files in data/ into the database. Idempotent."

    @transaction.atomic
    def handle(self, *args, **options):
        data_dir = settings.DATA_DIR
        skipped = []

        products, stock = [], {}
        for line, r in enumerate(_rows(data_dir / "catalogue.csv"), start=2):
            sku, price = (r.get("sku") or "").strip(), _money(r.get("price"))
            if not sku or price is None:
                skipped.append(f"catalogue.csv:{line} sku={sku!r} price={r.get('price')!r}")
                continue
            try:
                qty = max(0, int(str(r.get("stock", "0")).strip() or 0))
            except ValueError:
                qty = 0
            stock[sku] = qty
            products.append(Product(
                sku=sku, name=(r.get("name") or "").strip(), category=(r.get("category") or "").strip(),
                brand=(r.get("brand") or "").strip(), vehicle_fitment=(r.get("vehicle_fitment") or "").strip(),
                price=price, description=(r.get("description") or "").strip()))
        Product.objects.bulk_create(
            products, batch_size=500, update_conflicts=True, unique_fields=["sku"],
            update_fields=["name", "category", "brand", "vehicle_fitment", "price", "description", "updated_at"])
        ids = dict(Product.objects.filter(sku__in=stock).values_list("sku", "id"))
        # ignore_conflicts: existing inventory rows (live stock) are left exactly as they are.
        Inventory.objects.bulk_create(
            [Inventory(product_id=ids[sku], quantity=q) for sku, q in stock.items() if sku in ids],
            batch_size=500, ignore_conflicts=True)

        dealers = []
        for line, r in enumerate(_rows(data_dir / "dealers.csv"), start=2):
            code, credit = (r.get("dealer_code") or "").strip(), _money(r.get("credit_limit") or "0", allow_zero=True)
            if not code or credit is None:
                skipped.append(f"dealers.csv:{line} code={code!r}")
                continue
            dealers.append(Dealer(
                dealer_code=code, name=(r.get("name") or "").strip(), city=(r.get("city") or "").strip(),
                phone=(r.get("phone") or "").strip(), email=(r.get("email") or "").strip(),
                gstin=(r.get("gstin") or "").strip(), credit_limit=credit))
        Dealer.objects.bulk_create(
            dealers, update_conflicts=True, unique_fields=["dealer_code"],
            update_fields=["name", "city", "phone", "email", "gstin", "credit_limit"])

        for s in skipped:
            self.stderr.write(self.style.WARNING(f"skipped {s}"))
        self.stdout.write(self.style.SUCCESS(
            f"seed_data: {Product.objects.count()} products, {Dealer.objects.count()} dealers "
            f"({len(skipped)} rows skipped)"))
