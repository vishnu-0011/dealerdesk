from django.urls import path

from . import views

urlpatterns = [
    path("orders/", views.OrderListCreate.as_view()),
    path("orders/<str:order_number>/", views.OrderDetail.as_view()),
]
