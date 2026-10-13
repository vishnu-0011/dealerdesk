"""DealerDesk settings. Everything environment-specific comes from environment variables.

Required: DATABASE_URL (PostgreSQL). Also read: SUPPLIER_API_URL, SUPPLIER_API_KEY and one LLM key
(GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY or GROQ_API_KEY).
For local development you can put them in a .env file at the repo root (never commit it).
"""
import os
from pathlib import Path

import dj_database_url
from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")  # real environment variables always win over .env

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "core",
    "catalog",
    "orders",
    "assistant",
    "forecasting",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "dealerdesk.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "dealerdesk.wsgi.application"

# ---- Database: PostgreSQL only -------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ImproperlyConfigured("Set DATABASE_URL to a PostgreSQL URL, e.g. postgres://user:pass@localhost:5432/dealerdesk")
DATABASES = {"default": dj_database_url.parse(DATABASE_URL, conn_max_age=0)}
if DATABASES["default"]["ENGINE"] != "django.db.backends.postgresql":
    raise ImproperlyConfigured("DealerDesk requires PostgreSQL. SQLite can't test the concurrency rules.")

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---- API: no authentication in this assignment ----------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "UNAUTHENTICATED_USER": None,
    # JSON only: no browsable HTML API, and every error uses the contract's error envelope.
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
    "EXCEPTION_HANDLER": "core.errors.exception_handler",
    "DEFAULT_PAGINATION_CLASS": "core.pagination.StandardPagination",
    "PAGE_SIZE": 20,
}

# ---- Assignment settings --------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SUPPLIER_API_URL = os.environ.get("SUPPLIER_API_URL", "http://127.0.0.1:8099")
SUPPLIER_API_KEY = os.environ.get("SUPPLIER_API_KEY", "")

# ---- Dealer assistant ------------------------------------------------------------------
# LLM_PROVIDER picks the primary provider (groq | gemini | openai | anthropic). If unset, the
# first provider with a key is primary (order: groq, gemini, openai, anthropic). Providers
# that have a key and are not primary act as fallbacks when the primary errors or rate-limits.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").strip().lower()
LLM_MODELS = {
    "groq": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
    "gemini": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    "openai": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5"),
}
# Groq retires models without much notice (llama-3.3-70b-versatile went away on 2026-08-16), so a
# second Groq model is tried automatically when the first errors.
GROQ_FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "qwen/qwen3.6-27b")
ASSISTANT_REQUEST_BUDGET_SECONDS = float(os.environ.get("ASSISTANT_REQUEST_BUDGET_SECONDS", "48"))
ASSISTANT_HISTORY_MESSAGES = 10

# ---- Forecasting ------------------------------------------------------------------------
FORECAST_CSV = BASE_DIR / "ml" / "forecast.csv"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {"assistant": {"handlers": ["console"], "level": "INFO"},
                "core": {"handlers": ["console"], "level": "INFO"}},
}
