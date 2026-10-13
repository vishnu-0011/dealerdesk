"""Output-side checks: nothing the assistant says about SKUs, prices, stock or orders may go beyond
what this turn's tool calls returned. Violations make the agent retry, then fall back to a
templated reply built straight from tool data."""
from __future__ import annotations

import re
from decimal import Decimal

from catalog.models import Product
from orders.models import Order

from .tools import TurnState, _dec

SKU_RE = re.compile(r"\b[A-Z]{3}-\d{4}\b")
ORDER_RE = re.compile(r"\bORD-\d{8}-\d{4}\b")
MONEY_RES = [re.compile(r"(?:₹|Rs\.?|INR)\s*(\d[\d,]*(?:\.\d+)?)", re.I),
             re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:rupees|INR)\b", re.I)]
COUNT_RES = [re.compile(r"(\d[\d,]*)\s*(?:units?|pcs|pieces|nos)\b", re.I),
             re.compile(r"(\d[\d,]*)\s+(?:in stock|available|left|remaining|on hand)\b", re.I),
             re.compile(r"\b(?:stock|available|availability|inventory)(?:\s+(?:level|count|is|of|currently|now))*\s*(?:is|:|=|of|-)?\s*(\d[\d,]*)\b", re.I)]
DISCOUNT_RE = re.compile(r"\b\d{1,3}\s?%\s?(?:off|discount)\b|\b[A-Z]{3,}-[A-Z]{2,}-\d+\b", re.I)
CONFIRM_RE = re.compile(r"\b(?:order|it|draft)\s+(?:has been|was|is now|is)\s+(?:confirmed|approved|dispatched|shipped)\b|"
                        r"\bI(?:'ve| have)\s+(?:confirmed|approved|dispatched)\b|\bstock\s+(?:is|has been)\s+reserved\b", re.I)
NEGATION_RE = re.compile(r"\b(?:can(?:'|no)t|cannot|can not|unable|don't|do not|won't|not able|isn't|is not)\b", re.I)


def _strip_ids(text: str) -> str:
    return ORDER_RE.sub(" ", SKU_RE.sub(" ", text))


def verify_reply(reply: str, state: TurnState, user_message: str) -> list[str]:
    issues: list[str] = []
    skus = set(SKU_RE.findall(reply))
    if skus:
        real = set(Product.objects.filter(sku__in=skus).values_list("sku", flat=True))
        if skus - real:
            issues.append(f"SKU(s) that do not exist: {', '.join(sorted(skus - real))}. Never invent SKUs.")
        user_skus = set(SKU_RE.findall(user_message.upper()))
        stale = (skus & real) - state.skus_seen - user_skus
        if stale:
            issues.append(f"SKU(s) {', '.join(sorted(stale))} were not returned by a tool this turn. "
                          "Call search_parts/check_stock again before mentioning them.")
    orders = set(ORDER_RE.findall(reply))
    if orders - state.order_numbers:
        known = set(Order.objects.filter(order_number__in=orders, dealer=state.dealer).values_list("order_number", flat=True))
        if orders - state.order_numbers - known:
            issues.append("Order number(s) that were not created for this dealer. Only quote order numbers from create_draft_order.")

    body = _strip_ids(reply)
    money = {_dec(m) for rx in MONEY_RES for m in rx.findall(body)}
    counts = {_dec(m) for rx in COUNT_RES for m in rx.findall(body)}
    qtys = state.user_numbers | state.counts
    allowed_money = set(state.amounts) | state.user_numbers
    for price in list(state.amounts):                       # price x quantity the dealer or an order mentioned
        for q in qtys:
            if q == q.to_integral_value() and 0 < q <= 100000:
                allowed_money.add((price * q).quantize(Decimal("0.01")))
    bad_money = sorted(m for m in money if m not in allowed_money and m.quantize(Decimal("0.01")) not in allowed_money)
    bad_counts = sorted(c for c in counts if c not in qtys)
    if (money or counts) and not state.tool_calls:
        issues.append("You quoted prices/stock without calling a tool this turn. Call search_parts or check_stock first.")
    elif bad_money or bad_counts:
        bad = [f"₹{m}" for m in bad_money] + [f"{c} units" for c in bad_counts]
        issues.append("These figures do not match tool results from this turn: " + ", ".join(bad) +
                      ". Quote prices and stock exactly as the tools returned them.")
    if DISCOUNT_RE.search(reply) and not NEGATION_RE.search(reply):
        issues.append("You mentioned a discount/promo code. No discounts exist in tool data; remove it.")
    if CONFIRM_RE.search(reply) and not NEGATION_RE.search(reply):
        issues.append("You claimed an order was confirmed/dispatched or stock reserved. You can only create DRAFT orders, which hold no stock.")
    return issues


def fallback_reply(state: TurnState) -> str:
    """A reply assembled only from tool output, used when the LLM cannot produce a grounded one."""
    for entry in reversed(state.last_results):
        tool, r = entry["tool"], entry["result"]
        status = r.get("status")
        if tool == "create_draft_order" and status == "created":
            lines = "; ".join(f"{i['quantity']} x {i['name']} ({i['sku']})" for i in r["items"])
            return (f"I created draft order {r['order_number']}: {lines}. Total ₹{r['grand_total']} incl. GST. "
                    "It is a draft and does not reserve stock until the team confirms it.")
        if tool == "create_draft_order" and status == "not_created" and r.get("shortfalls"):
            s = r["shortfalls"][0]
            return (f"I couldn't create that order: only {s['available']} units of {s['name']} ({s['sku']}) are in stock "
                    f"and you asked for {s['requested']}. Want a draft for the available quantity instead?")
        if status == "ok" and r.get("results"):
            rows = "\n".join(f"- {i['name']} ({i['sku']}): ₹{i['price']}, {i['stock']} units in stock" for i in r["results"])
            return "Here is what I found:\n" + rows
        if tool == "check_stock" and status == "ok":
            return f"{r['name']} ({r['sku']}): ₹{r['price']}, {r['stock']} units in stock."
        if status in ("needs_vehicle", "ambiguous_vehicle"):
            opts = ", ".join(r.get("candidate_vehicles", [])[:8])
            return f"Which vehicle is this for? I have parts for: {opts}."
        if status == "unknown_vehicle":
            return "I don't have parts listed for that vehicle. I cover: " + ", ".join(r.get("supported_vehicles", [])) + "."
        if status == "needs_part_type":
            return "Which part do you need? Available for that vehicle: " + ", ".join(r.get("available_part_types", [])) + "."
        if status == "no_results":
            return "I couldn't find a matching part in the catalogue. Could you describe the part and vehicle another way?"
    return ("I can help you find parts, check live stock and prices, and create draft orders. "
            "What part and vehicle are you looking for?")
