from django.urls import path

from . import views

urlpatterns = [
    path("forecast/<str:sku>/", views.ForecastDetail.as_view()),
    path("reorder-suggestions/", views.ReorderSuggestions.as_view()),
]
