"""Order business rules shared by the REST API and the dealer assistant."""
from decimal import ROUND_HALF_UP, Decimal
from zoneinfo import ZoneInfo

from django.db import transaction
from django.utils import timezone

from catalog.models import Dealer, Product
from core.errors import ValidationFailed
from core.validation import fmt_money, require_object, strict_int

from .models import DailyOrderCounter, Order, OrderItem, Status

IST = ZoneInfo("Asia/Kolkata")
GST_RATE = Decimal("0.18")
CENT = Decimal("0.01")


def to_paisa(amount: Decimal) -> Decimal:
    """Round to 2 decimals, half-up (0.005 -> 0.01). Decimal's default is half-EVEN, which would
    turn 0.125 into 0.12 and drift from the accountant's numbers."""
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def price_line(unit_price: Decimal, quantity: int) -> dict:
    """GST is computed per line, never on the order total."""
    subtotal = to_paisa(unit_price * quantity)
    tax = to_paisa(subtotal * GST_RATE)
    return {"line_subtotal": subtotal, "line_tax": tax, "line_total": subtotal + tax}


def next_order_number() -> str:
    """ORD-YYYYMMDD-NNNN with the IST date. Must be called inside transaction.atomic()."""
    today = timezone.now().astimezone(IST).date()
    DailyOrderCounter.objects.bulk_create([DailyOrderCounter(date=today)], ignore_conflicts=True)
    counter = DailyOrderCounter.objects.select_for_update().get(date=today)
    counter.last_seq += 1
    counter.save(update_fields=["last_seq"])
    return f"ORD-{today:%Y%m%d}-{counter.last_seq:04d}"


def parse_items(raw) -> list[tuple[str, int]]:
    """Validate the `items` array. Returns [(sku, quantity)] in the given order."""
    if not isinstance(raw, list) or not raw:
        raise ValidationFailed("items must be a non-empty list.", [{"field": "items", "message": "at least one item"}])
    seen, parsed = set(), []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValidationFailed(f"items[{i}] must be an object.")
        sku = item.get("sku")
        if not isinstance(sku, str) or not sku.strip():
            raise ValidationFailed(f"items[{i}].sku is required.", [{"field": f"items[{i}].sku", "message": "required"}])
        sku = sku.strip()
        qty = strict_int(item.get("quantity"), f"items[{i}].quantity", minimum=1)
        if sku in seen:
            raise ValidationFailed(f"SKU {sku} appears more than once.", [{"field": "items", "message": f"duplicate sku {sku}"}])
        seen.add(sku)
        parsed.append((sku, qty))
    return parsed


def resolve_dealer(dealer_id) -> Dealer:
    dealer_id = strict_int(dealer_id, "dealer_id", minimum=1) if dealer_id is not None else None
    if dealer_id is None:
        raise ValidationFailed("dealer_id is required.", [{"field": "dealer_id", "message": "required"}])
    try:
        return Dealer.objects.get(pk=dealer_id)
    except Dealer.DoesNotExist:
        raise ValidationFailed(f"Dealer {dealer_id} does not exist.", [{"field": "dealer_id", "message": "unknown dealer"}])


def resolve_products(parsed) -> dict[str, Product]:
    products = {p.sku: p for p in Product.objects.select_related("inventory").filter(sku__in=[s for s, _ in parsed])}
    unknown = [s for s, _ in parsed if s not in products]
    if unknown:
        raise ValidationFailed("Unknown SKU: " + ", ".join(unknown),
                               [{"field": "items", "message": f"unknown sku {s}"} for s in unknown])
    return products


def create_draft_order(dealer: Dealer, parsed: list[tuple[str, int]]) -> Order:
    """Create a DRAFT at the current catalogue prices. A draft never touches stock."""
    products = resolve_products(parsed)
    with transaction.atomic():
        order = Order.objects.create(dealer=dealer, status=Status.DRAFT, order_number=next_order_number())
        subtotal = tax = Decimal("0.00")
        for sku, qty in parsed:
            product = products[sku]
            line = price_line(product.price, qty)
            OrderItem.objects.create(order=order, product=product, quantity=qty, unit_price=product.price, **line)
            subtotal += line["line_subtotal"]
            tax += line["line_tax"]
        order.subtotal, order.tax_total, order.grand_total = subtotal, tax, subtotal + tax
        order.save(update_fields=["subtotal", "tax_total", "grand_total", "updated_at"])
    return order


def create_draft_from_payload(body) -> Order:
    """Validation + creation for POST /api/orders/ (and anything else holding a raw payload)."""
    body = require_object(body)
    dealer = resolve_dealer(body.get("dealer_id"))
    return create_draft_order(dealer, parse_items(body.get("items")))


def order_dict(order: Order) -> dict:
    """Serialise an order. A DRAFT follows the *current* catalogue price (re-priced on read);
    every other status shows the stored, locked amounts."""
    items, subtotal, tax = [], Decimal("0.00"), Decimal("0.00")
    live = order.status == Status.DRAFT
    for it in order.items.all():
        if live:
            unit, line = it.product.price, price_line(it.product.price, it.quantity)
        else:
            unit = it.unit_price
            line = {"line_subtotal": it.line_subtotal, "line_tax": it.line_tax, "line_total": it.line_total}
        subtotal += line["line_subtotal"]
        tax += line["line_tax"]
        items.append({"sku": it.product.sku, "name": it.product.name, "quantity": it.quantity,
                      "unit_price": fmt_money(unit), **{k: fmt_money(v) for k, v in line.items()}})
    if not live:
        subtotal, tax = order.subtotal, order.tax_total
    return {"order_number": order.order_number, "dealer_id": order.dealer_id, "status": order.status,
            "items": items, "subtotal": fmt_money(subtotal), "tax_total": fmt_money(tax),
            "grand_total": fmt_money(subtotal + tax), "created_at": _iso(order.created_at),
            "updated_at": _iso(order.updated_at)}


def _iso(dt) -> str:
    return timezone.localtime(dt, IST).isoformat()
