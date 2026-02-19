"""Optional OpenLIT bootstrap for OpenTulpa runtime observability."""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
_initialized = False


def _as_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _set_default_env(var: str, value: str) -> None:
    if not os.environ.get(var):
        os.environ[var] = value


def init_openlit_from_env() -> None:
    """Initialize OpenLIT if enabled; never fail app startup."""
    global _initialized
    if _initialized:
        return
    if not _as_bool(os.environ.get("OPENLIT_ENABLED"), default=False):
        return

    _set_default_env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    _set_default_env("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    _set_default_env("OTEL_SERVICE_NAME", "opentulpa")
    _set_default_env("OTEL_DEPLOYMENT_ENVIRONMENT", "development")
    _set_default_env("OPENLIT_APPLICATION_NAME", os.environ.get("OTEL_SERVICE_NAME", "opentulpa"))

    try:
        import openlit  # type: ignore
    except Exception as exc:
        logger.warning("OpenLIT enabled but import failed: %s", exc)
        return

    kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(openlit.init)
    except Exception:
        sig = None
    if sig is not None:
        params = set(sig.parameters.keys())
        if "otlp_endpoint" in params and os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            kwargs["otlp_endpoint"] = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        if "application_name" in params and os.environ.get("OPENLIT_APPLICATION_NAME"):
            kwargs["application_name"] = os.environ["OPENLIT_APPLICATION_NAME"]
        if "environment" in params and os.environ.get("OTEL_DEPLOYMENT_ENVIRONMENT"):
            kwargs["environment"] = os.environ["OTEL_DEPLOYMENT_ENVIRONMENT"]
        if "service_name" in params and os.environ.get("OTEL_SERVICE_NAME"):
            kwargs["service_name"] = os.environ["OTEL_SERVICE_NAME"]

    try:
        if kwargs:
            openlit.init(**kwargs)
        else:
            openlit.init()
    except TypeError:
        # Fallback across OpenLIT versions.
        openlit.init()
    except Exception as exc:
        logger.warning("OpenLIT initialization failed: %s", exc)
        return

    logger.info(
        "OpenLIT initialized (endpoint=%s, service=%s)",
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        os.environ.get("OTEL_SERVICE_NAME", ""),
    )
    _initialized = True

