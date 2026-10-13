"""Diagnose the LLM connection:  python manage.py check_llm"""
from django.core.management.base import BaseCommand

from assistant.llm import LLMError, configured_providers
from assistant.tools import TOOL_SCHEMAS


class Command(BaseCommand):
    help = "Test the configured LLM provider(s) with a plain call and a tool-calling call."

    def handle(self, *args, **options):
        providers = configured_providers()
        if not providers:
            self.stdout.write(self.style.ERROR(
                "No provider key found. Set GROQ_API_KEY (or GEMINI/OPENAI/ANTHROPIC) in .env or the environment."))
            return
        self.stdout.write("Providers, in order of use: " + ", ".join(f"{p.name} ({p.model})" for p in providers))
        for p in providers:
            self._try(p, "plain call", [{"role": "user", "content": "Reply with the single word OK."}], None)
            self._try(p, "tool call", [{"role": "user", "content": "Find brake pads for a KTM Duke 390."}], TOOL_SCHEMAS)

    def _try(self, p, label, messages, tools):
        try:
            r = p.chat(messages, tools, timeout=30)
        except LLMError as e:
            self.stdout.write(self.style.ERROR(f"  [FAIL] {p.name} {label}: {e}"))
            return
        what = f"tool calls: {[c.name for c in r.tool_calls]}" if r.tool_calls else f"text: {r.content[:60]!r}"
        self.stdout.write(self.style.SUCCESS(f"  [ OK ] {p.name} {label} -> {what}"))
