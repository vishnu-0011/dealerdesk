from django.contrib import admin
from django.urls import include, path

handler404 = "core.errors.not_found_view"
handler500 = "core.errors.server_error_view"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("core.urls")),
    path("api/", include("catalog.urls")),
    path("api/", include("orders.urls")),
    path("api/", include("assistant.urls")),
    path("api/", include("forecasting.urls")),
]
