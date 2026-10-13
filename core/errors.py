"""One error envelope for the whole API:  {"error": {"code", "message", "details": []}}."""
import logging

from django.http import Http404, JsonResponse
from rest_framework import exceptions as drf_exc
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_default_handler

log = logging.getLogger("core")


class APIError(Exception):
    """Raise anywhere in a view/service to produce a contract-shaped error response."""

    def __init__(self, code: str, message: str, status: int = 400, details=None):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.details = details if details is not None else []


def ValidationFailed(message: str, details=None) -> APIError:  # noqa: N802 (reads like a class)
    return APIError("VALIDATION_ERROR", message, 400, details)


def NotFound(message: str = "Not found.") -> APIError:  # noqa: N802
    return APIError("NOT_FOUND", message, 404)


def Duplicate(message: str) -> APIError:  # noqa: N802
    return APIError("DUPLICATE", message, 409)


def envelope(code, message, details=None):
    return {"error": {"code": code, "message": message, "details": details if details is not None else []}}


def _flatten(detail, prefix=""):
    """DRF's nested ValidationError detail -> flat [{"field": ..., "message": ...}]."""
    out = []
    if isinstance(detail, dict):
        for k, v in detail.items():
            out += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(detail, (list, tuple)):
        for i, v in enumerate(detail):
            if isinstance(v, (dict, list, tuple)):
                out += _flatten(v, f"{prefix}[{i}]" if isinstance(v, dict) else prefix)
            else:
                out.append({"field": prefix, "message": str(v)})
    else:
        out.append({"field": prefix, "message": str(detail)})
    return out


def exception_handler(exc, context):
    if isinstance(exc, APIError):
        return Response(envelope(exc.code, exc.message, exc.details), status=exc.status)
    if isinstance(exc, Http404):
        return Response(envelope("NOT_FOUND", "Not found."), status=404)
    if isinstance(exc, drf_exc.ValidationError):
        details = _flatten(exc.detail)
        return Response(envelope("VALIDATION_ERROR", "Invalid input.", details), status=400)
    if isinstance(exc, drf_exc.ParseError):
        return Response(envelope("VALIDATION_ERROR", "Request body is not valid JSON."), status=400)
    if isinstance(exc, drf_exc.NotFound):
        return Response(envelope("NOT_FOUND", str(exc.detail)), status=404)
    if isinstance(exc, drf_exc.MethodNotAllowed):
        return Response(envelope("METHOD_NOT_ALLOWED", str(exc.detail)), status=405)
    if isinstance(exc, drf_exc.UnsupportedMediaType):
        return Response(envelope("VALIDATION_ERROR", "Send the body as application/json."), status=415)
    response = drf_default_handler(exc, context)
    if response is not None:
        return Response(envelope("ERROR", str(getattr(exc, "detail", exc))), status=response.status_code)
    log.exception("Unhandled error in %s", context.get("view"))
    return Response(envelope("INTERNAL_ERROR", "Unexpected server error."), status=500)


def not_found_view(request, exception=None):
    return JsonResponse(envelope("NOT_FOUND", "Not found."), status=404)


def server_error_view(request):
    return JsonResponse(envelope("INTERNAL_ERROR", "Unexpected server error."), status=500)
