from django.urls import path

from . import views

urlpatterns = [
    path("products/", views.ProductListCreate.as_view()),
    path("products/<str:sku>/", views.ProductDetail.as_view()),
    path("inventory/<str:sku>/", views.InventoryDetail.as_view()),
    path("dealers/", views.DealerListCreate.as_view()),
    path("dealers/<int:pk>/", views.DealerDetail.as_view()),
]
