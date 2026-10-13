import time

from django.conf import settings
from rest_framework.response import Response
from rest_framework.views import APIView

from core.errors import ValidationFailed
from core.validation import require_object, text_value
from orders.services import resolve_dealer

from . import agent, llm
from .models import ChatSession

MAX_MESSAGE_CHARS = 2000


def build_router() -> llm.Router:
    """Separate function so tests can swap in a scripted LLM."""
    deadline = time.monotonic() + settings.ASSISTANT_REQUEST_BUDGET_SECONDS
    return llm.Router(llm.configured_providers(), deadline)


class ChatView(APIView):
    def post(self, request):
        body = require_object(request.data)
        session_id = text_value(body.get("session_id"), "session_id", required=True, max_length=100)
        message = text_value(body.get("message"), "message", required=True, max_length=MAX_MESSAGE_CHARS)
        dealer = resolve_dealer(body.get("dealer_id"))
        session, _ = ChatSession.objects.get_or_create(session_id=session_id, dealer=dealer)
        return Response(agent.run_turn(session, dealer, message, build_router()))
