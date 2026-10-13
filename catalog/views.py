from django.core.validators import validate_email
from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import IntegrityError, transaction
from django.db.models import Q
from rest_framework.response import Response
from rest_framework.views import APIView

from core.errors import Duplicate, NotFound, ValidationFailed
from core.pagination import StandardPagination
from core.validation import fmt_money, money_value, require_object, strict_int, text_value

from .models import Dealer, Inventory, InventoryAdjustment, Product

PRODUCT_EDITABLE = ("name", "category", "brand", "vehicle_fitment", "price", "description")


def product_dict(p: Product) -> dict:
    return {"sku": p.sku, "name": p.name, "category": p.category, "brand": p.brand,
            "vehicle_fitment": p.vehicle_fitment, "price": fmt_money(p.price), "stock": p.stock,
            "description": p.description}


def dealer_dict(d: Dealer) -> dict:
    return {"id": d.id, "dealer_code": d.dealer_code, "name": d.name, "city": d.city, "phone": d.phone,
            "email": d.email, "gstin": d.gstin, "credit_limit": fmt_money(d.credit_limit)}


def get_product_or_404(sku: str) -> Product:
    try:
        return Product.objects.select_related("inventory").get(sku=sku)
    except Product.DoesNotExist:
        raise NotFound(f"Product {sku} not found.")


def paginate(request, queryset, serializer):
    paginator = StandardPagination()
    page = paginator.paginate_queryset(queryset, request)
    return paginator.get_paginated_response([serializer(o) for o in page])


class ProductListCreate(APIView):
    def get(self, request):
        qs = Product.objects.select_related("inventory").order_by("sku")
        q = request.query_params
        if q.get("search"):
            term = q["search"].strip()
            qs = qs.filter(Q(sku__icontains=term) | Q(name__icontains=term) | Q(brand__icontains=term))
        if q.get("category"):
            qs = qs.filter(category=q["category"])
        if q.get("vehicle"):
            qs = qs.filter(vehicle_fitment__icontains=q["vehicle"].strip())
        return paginate(request, qs, product_dict)

    def post(self, request):
        body = require_object(request.data)
        sku = text_value(body.get("sku"), "sku", required=True, max_length=40)
        name = text_value(body.get("name"), "name", required=True)
        price = money_value(body.get("price"), "price")
        fields = dict(
            sku=sku, name=name, price=price,
            category=text_value(body.get("category"), "category", max_length=100),
            brand=text_value(body.get("brand"), "brand", max_length=100),
            vehicle_fitment=text_value(body.get("vehicle_fitment"), "vehicle_fitment", max_length=150),
            description=text_value(body.get("description"), "description", max_length=5000),
        )
        if Product.objects.filter(sku=sku).exists():
            raise Duplicate(f"A product with SKU {sku} already exists.")
        try:
            with transaction.atomic():
                product = Product.objects.create(**fields)
                Inventory.objects.create(product=product, quantity=0)
        except IntegrityError:  # lost a race with a concurrent create of the same SKU
            raise Duplicate(f"A product with SKU {sku} already exists.")
        return Response(product_dict(get_product_or_404(sku)), status=201)


class ProductDetail(APIView):
    def get(self, request, sku):
        return Response(product_dict(get_product_or_404(sku)))

    def patch(self, request, sku):
        product = get_product_or_404(sku)
        body = require_object(request.data)
        if "sku" in body and body["sku"] != product.sku:
            raise ValidationFailed("SKU cannot be changed.", [{"field": "sku", "message": "read-only"}])
        updates = {}
        for field in PRODUCT_EDITABLE:
            if field not in body:
                continue
            if field == "price":
                updates[field] = money_value(body[field], "price")
            elif field == "name":
                updates[field] = text_value(body[field], "name", required=True)
            elif field == "description":
                updates[field] = text_value(body[field], "description", max_length=5000)
            else:
                updates[field] = text_value(body[field], field, max_length=150)
        if updates:
            for k, v in updates.items():
                setattr(product, k, v)
            product.save(update_fields=[*updates, "updated_at"])
        return Response(product_dict(get_product_or_404(sku)))


class InventoryDetail(APIView):
    def put(self, request, sku):
        body = require_object(request.data)
        quantity = strict_int(body.get("quantity"), "quantity", minimum=0)
        reason = text_value(body.get("reason"), "reason")
        with transaction.atomic():
            try:
                product = Product.objects.get(sku=sku)
            except Product.DoesNotExist:
                raise NotFound(f"Product {sku} not found.")
            inv, _ = Inventory.objects.select_for_update().get_or_create(product=product)
            InventoryAdjustment.objects.create(product=product, old_quantity=inv.quantity,
                                               new_quantity=quantity, reason=reason)
            inv.quantity = quantity
            inv.save(update_fields=["quantity", "updated_at"])
        return Response({"sku": sku, "quantity": quantity})


class DealerListCreate(APIView):
    def get(self, request):
        return paginate(request, Dealer.objects.order_by("id"), dealer_dict)

    def post(self, request):
        body = require_object(request.data)
        code = text_value(body.get("dealer_code"), "dealer_code", required=True, max_length=40)
        name = text_value(body.get("name"), "name", required=True)
        email = text_value(body.get("email"), "email", max_length=254)
        if email:
            try:
                validate_email(email)
            except DjangoValidationError:
                raise ValidationFailed("email is not valid.", [{"field": "email", "message": "invalid email"}])
        credit = money_value(body["credit_limit"], "credit_limit", allow_zero=True) if body.get("credit_limit") is not None else 0
        if Dealer.objects.filter(dealer_code=code).exists():
            raise Duplicate(f"A dealer with code {code} already exists.")
        try:
            dealer = Dealer.objects.create(
                dealer_code=code, name=name, email=email, credit_limit=credit,
                city=text_value(body.get("city"), "city", max_length=100),
                phone=text_value(body.get("phone"), "phone", max_length=30),
                gstin=text_value(body.get("gstin"), "gstin", max_length=20))
        except IntegrityError:
            raise Duplicate(f"A dealer with code {code} already exists.")
        return Response(dealer_dict(dealer), status=201)


class DealerDetail(APIView):
    def get(self, request, pk):
        try:
            return Response(dealer_dict(Dealer.objects.get(pk=pk)))
        except Dealer.DoesNotExist:
            raise NotFound(f"Dealer {pk} not found.")
