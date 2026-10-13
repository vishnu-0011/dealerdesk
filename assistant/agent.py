"""One assistant turn: LLM <-> tools loop, then verification of the final reply.

    user message -> [LLM -> tool calls -> tool results]* -> draft reply -> grounding check
        -> ok: return it            -> not ok: tell the LLM what is wrong, retry (max 2)
        -> still not ok / LLM down: reply built directly from tool output (grounding.fallback_reply)
"""
from __future__ import annotations

import logging

from django.conf import settings
from django.db import transaction

from catalog.models import Dealer, Product

from .grounding import SKU_RE, fallback_reply, verify_reply
from .llm import LLMError, Router
from .models import ChatMessage, ChatSession
from .prompts import CORRECTION, system_prompt
from .tools import TOOL_SCHEMAS, TurnState, run_tool

log = logging.getLogger("assistant")
MAX_LLM_CALLS = 7
MAX_CORRECTIONS = 2
UNAVAILABLE = "I'm having trouble reaching my assistant service right now. Please try again in a moment."


def load_history(session: ChatSession) -> list[dict]:
    rows = list(session.messages.order_by("-id")[: settings.ASSISTANT_HISTORY_MESSAGES])
    return [{"role": m.role, "content": m.content} for m in reversed(rows)]


def cited_skus(reply: str) -> list[str]:
    seen = list(dict.fromkeys(SKU_RE.findall(reply)))
    real = set(Product.objects.filter(sku__in=seen).values_list("sku", flat=True))
    return [s for s in seen if s in real]


def run_turn(session: ChatSession, dealer: Dealer, message: str, router: Router) -> dict:
    state = TurnState(dealer, message)
    messages = [{"role": "system", "content": system_prompt(dealer.name)}, *load_history(session),
                {"role": "user", "content": message}]
    reply, llm_failed, corrections, tool_choice = None, False, 0, None

    for _ in range(MAX_LLM_CALLS):
        try:
            resp = router.chat(messages, TOOL_SCHEMAS, tool_choice)
        except LLMError as e:
            log.error("assistant LLM failure: %s", e)
            llm_failed = True
            break
        tool_choice = None
        if resp.tool_calls:
            messages.append(resp.as_message())
            for call in resp.tool_calls:
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": run_tool(state, call.name, call.arguments)})
            continue
        issues = verify_reply(resp.content, state, message) if resp.content else ["The reply was empty."]
        if not issues:
            reply = resp.content
            break
        log.info("grounding issues (correction %d): %s", corrections + 1, issues)
        if corrections >= MAX_CORRECTIONS:
            break
        corrections += 1
        messages.append(resp.as_message() if resp.content else {"role": "assistant", "content": "(empty)"})
        messages.append({"role": "user", "content": CORRECTION.format(issues=" ".join(issues))})
        if any("without calling a tool" in i or "not returned by a tool" in i for i in issues):
            tool_choice = "required"

    if reply is None:
        reply = UNAVAILABLE if (llm_failed and not state.last_results) else fallback_reply(state)

    with transaction.atomic():
        ChatMessage.objects.create(session=session, role="user", content=message)
        ChatMessage.objects.create(session=session, role="assistant", content=reply)
    return {"reply": reply, "cited_skus": cited_skus(reply), "tool_calls": state.tool_calls, "actions": state.actions}
