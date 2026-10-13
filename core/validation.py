"""Strict input parsing. DRF fields coerce ("3" -> 3, 2.0 -> 2); the contract wants rejection."""
import re
from decimal import Decimal, InvalidOperation

from .errors import ValidationFailed

_TWO_DP = re.compile(r"^\d+(\.\d{1,2})?$")


def require_object(body):
    if not isinstance(body, dict):
        raise ValidationFailed("Request body must be a JSON object.")
    return body


def strict_int(value, field: str, minimum: int | None = None) -> int:
    """Accept only real JSON integers (not bools, floats or numeric strings)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationFailed(f"{field} must be an integer.", [{"field": field, "message": "must be an integer"}])
    if minimum is not None and value < minimum:
        raise ValidationFailed(f"{field} must be >= {minimum}.", [{"field": field, "message": f"must be >= {minimum}"}])
    return value


def money_value(value, field: str = "price", allow_zero: bool = False) -> Decimal:
    """A positive (or, if allow_zero, non-negative) amount with at most 2 decimals."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValidationFailed(f"{field} must be a number.", [{"field": field, "message": "must be a number"}])
    text = value.strip() if isinstance(value, str) else format(Decimal(str(value)), "f")
    if not _TWO_DP.match(text):
        raise ValidationFailed(f"{field} must be a positive amount with at most 2 decimals.",
                               [{"field": field, "message": "positive, max 2 decimals"}])
    try:
        amount = Decimal(text)
    except InvalidOperation:
        raise ValidationFailed(f"{field} must be a number.")
    if amount < 0 or (amount == 0 and not allow_zero):
        raise ValidationFailed(f"{field} must be greater than 0.", [{"field": field, "message": "must be > 0"}])
    return amount


def text_value(value, field: str, required: bool = False, max_length: int = 255, default: str = "") -> str:
    if value is None:
        if required:
            raise ValidationFailed(f"{field} is required.", [{"field": field, "message": "required"}])
        return default
    if not isinstance(value, str):
        raise ValidationFailed(f"{field} must be a string.", [{"field": field, "message": "must be a string"}])
    value = value.strip()
    if required and not value:
        raise ValidationFailed(f"{field} is required.", [{"field": field, "message": "required"}])
    if len(value) > max_length:
        raise ValidationFailed(f"{field} is too long (max {max_length}).", [{"field": field, "message": "too long"}])
    return value


def fmt_money(amount: Decimal) -> str:
    return f"{amount:.2f}"
