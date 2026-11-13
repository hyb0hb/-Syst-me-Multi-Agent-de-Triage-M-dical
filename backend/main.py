"""Application principale FastAPI du backend MedTriage AI.

Architecture modulaire :
- Routers : endpoints REST modulaires (/api/triage, /api/presets, /api/metrics)
- Services : chargement des modèles, validation clinique, inférence
- Schemas : validation et documentation OpenAPI Pydantic
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import STATIC_DIR
from backend.routers import metrics_router, presets_router, triage_router
from backend.routers.history import router as history_router
from backend.services.model_service import load_metrics, warmup_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise le cache et réchauffe les modèles au démarrage du serveur."""
    load_metrics()
    warmup_models()
    yield


app = FastAPI(
    title="MedTriage AI — Multi-Agent Medical Triage Backend",
    description=(
        "Backend REST asynchrone haute performance orchestrant 3 agents d'IA : "
        "Agent 1 (Vision CNN MobileNetV2), Agent 2 (Clinique Random Forest) "
        "et Agent 3 (Consensus Régression Logistique)."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configuration CORS sécurisée (conforme W3C : pas de wildcard * avec allow_credentials=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routeurs spécialisés
app.include_router(triage_router)
app.include_router(presets_router)
app.include_router(metrics_router)
app.include_router(history_router)

# Montage des fichiers statiques (UI web)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", summary="Page d'accueil de l'interface utilisateur")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")
