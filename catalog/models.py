from django.db import models


class Product(models.Model):
    """A sellable part. `price` is the *current* catalogue price (order lines snapshot it)."""

    sku = models.CharField(max_length=40, unique=True)
    name = models.CharField(max_length=255)
    category = models.CharField(max_length=100, blank=True, db_index=True)
    brand = models.CharField(max_length=100, blank=True)
    vehicle_fitment = models.CharField(max_length=150, blank=True, db_index=True)
    price = models.DecimalField(max_digits=12, decimal_places=2)
    description = models.TextField(blank=True)  # supplier-written: untrusted free text
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["sku"]
        constraints = [models.CheckConstraint(condition=models.Q(price__gt=0), name="product_price_positive")]

    def __str__(self):
        return f"{self.sku} {self.name}"

    @property
    def stock(self) -> int:
        inv = getattr(self, "inventory", None)
        return inv.quantity if inv else 0


class Inventory(models.Model):
    """Available quantity, kept apart from Product so stock writes lock one narrow row and never
    contend with catalogue edits (price/name changes, supplier sync)."""

    product = models.OneToOneField(Product, on_delete=models.CASCADE, related_name="inventory")
    quantity = models.PositiveIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.CheckConstraint(condition=models.Q(quantity__gte=0), name="inventory_quantity_non_negative")]


class InventoryAdjustment(models.Model):
    """Audit trail for manual stock corrections (PUT /api/inventory/{sku}/)."""

    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="adjustments")
    old_quantity = models.PositiveIntegerField()
    new_quantity = models.PositiveIntegerField()
    reason = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Dealer(models.Model):
    dealer_code = models.CharField(max_length=40, unique=True)
    name = models.CharField(max_length=255)
    city = models.CharField(max_length=100, blank=True)
    phone = models.CharField(max_length=30, blank=True)
    email = models.CharField(max_length=254, blank=True)
    gstin = models.CharField(max_length=20, blank=True)
    credit_limit = models.DecimalField(max_digits=12, decimal_places=2, default=0)  # stored; no rule uses it yet
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return f"{self.dealer_code} {self.name}"
