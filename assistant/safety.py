"""Treat catalogue text as data. Two defences live here:

1. `looks_injected` / `sanitize_field`: text that reads like an instruction to an AI is removed
   before it can reach the LLM or the retrieval index.
2. The agent never sends product *descriptions* to the LLM at all (see tools.py), so the usual
   carrier of hidden instructions is simply not in the model's context.

These are heuristics. They stop the obvious phrasings, not a determined attacker; the output-side
checks in grounding.py are the backstop (see DECISIONS.md, answer 5).
"""
import re
import unicodedata

_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+|the\s+)*(?:previous|prior|above|earlier|preceding)\s+(?:instruction|prompt|rule|direction|message)",
    r"disregard\s+.{0,40}(?:instruction|rule|prompt|polic)",
    r"\bnotes?\s+(?:for|to)\s+(?:the\s+)?(?:ai|a\.i\.|assistants?|llms?|models?|chat\s?bots?|agents?)\b",
    r"\[\s*(?:system|assistant|developer|admin|root)\s*\]",
    r"<\s*/?\s*(?:system|assistant|developer|tool|instructions?)\s*>",
    r"\b(?:system|developer)\s+(?:prompt|message|instruction)",
    r"\byou\s+are\s+now\b",
    r"\b(?:admin|developer|debug|god)\s+mode\b",
    r"\bjail\s?break",
    r"\b(?:tell|instruct|inform|advise)\s+(?:the\s+)?(?:customer|user|dealer|buyer)s?\b",
    r"\bwithout\s+(?:checking|verifying)\s+(?:the\s+)?stock",
    r"\b(?:confirm|place|approve)\s+(?:the\s+|this\s+)?(?:dealer'?s\s+)?order\b",
    r"\b(?:when|if)\s+(?:asked|the\s+(?:user|dealer|customer)\s+asks)\b.{0,60}\b(?:say|reply|respond|tell|confirm)",
    r"\b\d{1,3}\s?%\s?off\b.{0,40}\b(?:code|today|everything|every)",
    r"\bpromo(?:tion(?:al)?)?\s+code\b",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE | re.DOTALL)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
REMOVED = "[removed]"


def normalise(text: str) -> str:
    """NFKC-fold look-alike characters, drop control/zero-width/bidi characters, collapse spaces."""
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", _CONTROL.sub("", text)).strip()


def looks_injected(text: str) -> bool:
    return bool(_INJECTION.search(normalise(text)))


def sanitize_field(text: str, max_len: int = 120) -> str:
    """For short catalogue fields shown to the LLM (name, brand, category, fitment)."""
    text = normalise(text)
    if looks_injected(text):
        return REMOVED
    return text[:max_len]


def strip_injected_sentences(text: str) -> str:
    """For free-text descriptions used only by the retrieval index: drop offending sentences."""
    sentences = re.split(r"(?<=[.!?])\s+|\s*\[[^\]]*\]\s*", normalise(text))
    return " ".join(s for s in sentences if s and not looks_injected(s))
