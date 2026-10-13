from django.db import models

from catalog.models import Dealer


class ChatSession(models.Model):
    """One conversation. Keyed by (session_id, dealer): the same session_id used by two different
    dealers is two different conversations, so memory can never leak across dealers."""

    session_id = models.CharField(max_length=100)
    dealer = models.ForeignKey(Dealer, on_delete=models.CASCADE, related_name="chat_sessions")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["session_id", "dealer"], name="unique_session_per_dealer")]


class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10)           # user | assistant
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]
