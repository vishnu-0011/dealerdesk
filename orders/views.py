from rest_framework.response import Response
from rest_framework.views import APIView

from catalog.views import paginate
from core.errors import NotFound, ValidationFailed

from .models import Order, Status
from .services import create_draft_from_payload, order_dict


def _with_items(qs):
    return qs.prefetch_related("items__product")


class OrderListCreate(APIView):
    def get(self, request):
        qs = _with_items(Order.objects.all())
        q = request.query_params
        if q.get("dealer_id") not in (None, ""):
            try:
                qs = qs.filter(dealer_id=int(q["dealer_id"]))
            except ValueError:
                raise ValidationFailed("dealer_id must be an integer.", [{"field": "dealer_id", "message": "integer"}])
        if q.get("status"):
            status = q["status"].upper()
            if status not in Status.values:
                raise ValidationFailed(f"status must be one of {', '.join(Status.values)}.")
            qs = qs.filter(status=status)
        return paginate(request, qs, order_dict)

    def post(self, request):
        order = create_draft_from_payload(request.data)
        return Response(order_dict(_with_items(Order.objects.filter(pk=order.pk)).get()), status=201)


class OrderDetail(APIView):
    def get(self, request, order_number):
        try:
            order = _with_items(Order.objects.all()).get(order_number=order_number)
        except Order.DoesNotExist:
            raise NotFound(f"Order {order_number} not found.")
        return Response(order_dict(order))
