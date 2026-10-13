"""The three tools the LLM may call, and the per-turn state that records what they returned.

Design rules:
* The LLM never supplies a dealer id: the dealer comes from the authenticated request, so a prompt
  cannot make the assistant act for someone else.
* Product descriptions are never returned. Only sanitised name/brand/category/fitment, live price
  and live stock are.
* `create_draft_order` goes through the same service as POST /api/orders/, so it has the same
  validation, pricing and numbering. It only ever creates DRAFTs.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal

from catalog.models import Dealer, Product
from core.errors import APIError
from core.validation import fmt_money
from orders import services as order_services

from . import retrieval
from .safety import sanitize_field

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_parts",
        "description": "Find parts in the catalogue. Returns live price and stock. Call this for any question "
                       "about what parts exist, what fits a vehicle, or what something costs.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The part wanted, e.g. 'brake pads', 'chain lube', 'LED headlight'."},
            "vehicle": {"type": "string", "description": "Make and model if the dealer named one, e.g. 'KTM Duke 390'. Omit if not given."},
            "category": {"type": "string", "description": "Optional catalogue category filter."},
            "limit": {"type": "integer", "description": "Max results (default 5)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "check_stock",
        "description": "Get the live stock level and current price of one SKU (e.g. BRK-1044).",
        "parameters": {"type": "object", "properties": {
            "sku": {"type": "string", "description": "The SKU, format AAA-1234."}}, "required": ["sku"]}}},
    {"type": "function", "function": {
        "name": "create_draft_order",
        "description": "Create a DRAFT order for the current dealer. Only call when the dealer has clearly asked to "
                       "order / put aside / reserve specific SKUs and quantities. Never confirms or reserves stock.",
        "parameters": {"type": "object", "properties": {
            "items": {"type": "array", "description": "Lines to order.", "items": {"type": "object", "properties": {
                "sku": {"type": "string"}, "quantity": {"type": "integer", "description": "Whole units, >= 1."}},
                "required": ["sku", "quantity"]}}}, "required": ["items"]}}},
]
MAX_TOOL_CALLS_PER_TURN = 8
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _dec(x) -> Decimal:
    return Decimal(str(x).replace(",", ""))


def numbers_in(text: str) -> set[Decimal]:
    return {_dec(m) for m in _NUM.findall(text or "")}


class TurnState:
    """Everything observable this turn. The grounding check reads from here."""

    def __init__(self, dealer: Dealer, user_message: str):
        self.dealer = dealer
        self.user_numbers = numbers_in(user_message)
        self.tool_calls: list[dict] = []
        self.skus_seen: set[str] = set()           # SKUs returned by a tool this turn
        self.amounts: set[Decimal] = set()         # prices / totals returned this turn
        self.counts: set[Decimal] = set()          # stock levels / quantities returned this turn
        self.order_numbers: set[str] = set()
        self.actions: list[dict] = []
        self.last_results: list[dict] = []         # (tool, result) kept for the fallback reply
        self._order_cache: dict = {}

    def record(self, name, args, result):
        self.tool_calls.append({"name": name, "arguments": args})
        self.last_results.append({"tool": name, "result": result})

    def money(self, v) -> str:
        d = _dec(v)
        self.amounts.add(d)
        return fmt_money(d)


def _item(p: Product, state: TurnState, fit: str | None = None) -> dict:
    state.skus_seen.add(p.sku)
    state.counts.add(Decimal(p.stock))
    out = {"sku": p.sku, "name": sanitize_field(p.name, 200), "brand": sanitize_field(p.brand),
           "category": sanitize_field(p.category), "vehicle_fitment": sanitize_field(p.vehicle_fitment) or "Universal",
           "price": state.money(p.price), "stock": p.stock}
    if fit:
        out["fit"] = fit
    return out


def search_parts(state: TurnState, query="", vehicle=None, category=None, limit=5, **_):
    found = retrieval.search(str(query or ""), vehicle and str(vehicle), category and str(category), limit)
    if found["status"] != "ok":
        return found
    skus = [s for s, _ in found["matches"]]
    products = {p.sku: p for p in Product.objects.select_related("inventory").filter(sku__in=skus)}  # live data
    results = [_item(products[s], state, fit) for s, fit in found["matches"] if s in products]
    return {"status": "ok", "vehicles": found.get("vehicles", []), "results": results}


def check_stock(state: TurnState, sku="", **_):
    sku = str(sku or "").strip().upper()
    try:
        p = Product.objects.select_related("inventory").get(sku=sku)
    except Product.DoesNotExist:
        return {"status": "unknown_sku", "sku": sku, "message": "No product with that SKU."}
    item = _item(p, state)
    item["in_stock"] = p.stock > 0
    return {"status": "ok", **item}


def create_draft_order(state: TurnState, items=None, **_):
    try:
        parsed = order_services.parse_items(items)
        products = order_services.resolve_products(parsed)
    except APIError as e:
        return {"status": "not_created", "reason": "invalid_request", "message": e.message}
    key = tuple(sorted(parsed))
    if key in state._order_cache:                          # the model retried the same call
        return {**state._order_cache[key], "note": "Duplicate call ignored; this order already exists."}
    short = [{"sku": s, "name": sanitize_field(products[s].name, 200), "requested": q, "available": products[s].stock}
             for s, q in parsed if q > products[s].stock]
    if short:
        for row in short:
            state.skus_seen.add(row["sku"])
            state.counts.add(Decimal(row["available"]))
        return {"status": "not_created", "reason": "insufficient_stock", "shortfalls": short,
                "message": "Not enough stock. Offer the available quantity instead; nothing was created."}
    order = order_services.create_draft_order(state.dealer, parsed)
    view = order_services.order_dict(order.__class__.objects.prefetch_related("items__product").get(pk=order.pk))
    state.order_numbers.add(view["order_number"])
    for line in view["items"]:
        state.skus_seen.add(line["sku"])
        state.counts.add(Decimal(line["quantity"]))
        for k in ("unit_price", "line_subtotal", "line_tax", "line_total"):
            state.amounts.add(_dec(line[k]))
    for k in ("subtotal", "tax_total", "grand_total"):
        state.amounts.add(_dec(view[k]))
    state.actions.append({"type": "draft_order_created", "order_number": view["order_number"]})
    result = {"status": "created", "order_number": view["order_number"], "order_status": "DRAFT",
              "items": view["items"], "subtotal": view["subtotal"], "tax_total": view["tax_total"],
              "grand_total": view["grand_total"], "stock_reserved": False,
              "note": "This is a DRAFT. It does not reserve stock; the team must confirm it."}
    state._order_cache[key] = result
    return result


TOOLS = {"search_parts": search_parts, "check_stock": check_stock, "create_draft_order": create_draft_order}


def run_tool(state: TurnState, name: str, arguments) -> str:
    """Execute one tool call; returns the JSON string handed back to the LLM."""
    args = arguments if isinstance(arguments, dict) else {}
    if len(state.tool_calls) >= MAX_TOOL_CALLS_PER_TURN:
        result = {"status": "error", "message": "Tool call limit reached for this turn."}
    elif name not in TOOLS:
        result = {"status": "error", "message": f"Unknown tool {name!r}. Use search_parts, check_stock or create_draft_order."}
    else:
        try:
            result = TOOLS[name](state, **args)
        except TypeError:
            result = {"status": "error", "message": "Invalid arguments."}
    state.record(name, args, result)
    return json.dumps(result, ensure_ascii=False)
