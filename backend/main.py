"""Application principale FastAPI du backend MedTriage AI.

Architecture modulaire :
- Routers : endpoints REST modulaires (/api/triage, /api/presets, /api/metrics)
- Services : chargement des modèles, validation clinique, inférence
- Schemas : validation et documentation OpenAPI Pydantic
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import STATIC_DIR
from backend.routers import metrics_router, presets_router, triage_router
from backend.routers.history import router as history_router
from backend.services.model_service import load_metrics, warmup_models

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
)

# Configuration CORS pour intégration frontend / clients tiers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.on_event("startup")
def startup_event():
    """Initialise le cache et réchauffe les modèles dès le lancement."""
    load_metrics()
    warmup_models()


@app.get("/", summary="Page d'accueil de l'interface utilisateur")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")
