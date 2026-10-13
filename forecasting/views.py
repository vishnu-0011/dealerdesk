from rest_framework.response import Response
from rest_framework.views import APIView

from core.errors import NotFound

from . import service


class ForecastDetail(APIView):
    def get(self, request, sku):
        forecast = service.forecast_for(sku)
        if forecast is None:
            raise NotFound(f"No forecast for SKU {sku}.")
        return Response({"sku": sku, "forecast": forecast})


class ReorderSuggestions(APIView):
    pagination_class = None  # one entry per forecast SKU, deliberately not paginated

    def get(self, request):
        return Response({"results": service.reorder_suggestions()})
