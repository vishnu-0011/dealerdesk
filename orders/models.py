from django.db import models

from catalog.models import Dealer, Product


class Status(models.TextChoices):
    DRAFT = "DRAFT"
    CONFIRMED = "CONFIRMED"
    DISPATCHED = "DISPATCHED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"


class Order(models.Model):
    order_number = models.CharField(max_length=20, unique=True)
    # PROTECT: a dealer with order history must never be deletable (accounting trail).
    dealer = models.ForeignKey(Dealer, on_delete=models.PROTECT, related_name="orders")
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.DRAFT, db_index=True)
    subtotal = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    tax_total = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    grand_total = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        indexes = [models.Index(fields=["dealer", "-created_at"])]


class OrderItem(models.Model):
    # CASCADE: lines have no meaning without their order.
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name="items")
    # PROTECT: a product that appears on an order can't be deleted from under it.
    product = models.ForeignKey(Product, on_delete=models.PROTECT, related_name="order_items")
    quantity = models.PositiveIntegerField()
    unit_price = models.DecimalField(max_digits=12, decimal_places=2)
    line_subtotal = models.DecimalField(max_digits=14, decimal_places=2)
    line_tax = models.DecimalField(max_digits=14, decimal_places=2)
    line_total = models.DecimalField(max_digits=14, decimal_places=2)

    class Meta:
        ordering = ["id"]
        constraints = [
            models.UniqueConstraint(fields=["order", "product"], name="one_line_per_product_per_order"),
            models.CheckConstraint(condition=models.Q(quantity__gte=1), name="orderitem_quantity_positive"),
        ]


class DailyOrderCounter(models.Model):
    """One row per IST day. Locking it (SELECT ... FOR UPDATE) hands out gapless, unique
    sequence numbers even when many orders are created at the same instant."""

    date = models.DateField(primary_key=True)
    last_seq = models.PositiveIntegerField(default=0)
