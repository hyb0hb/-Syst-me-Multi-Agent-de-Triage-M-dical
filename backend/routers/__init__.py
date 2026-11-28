"""Package des routeurs d'endpoints de l'API."""

from backend.routers.metrics import router as metrics_router
from backend.routers.presets import router as presets_router
from backend.routers.triage import router as triage_router

__all__ = ["triage_router", "presets_router", "metrics_router"]
