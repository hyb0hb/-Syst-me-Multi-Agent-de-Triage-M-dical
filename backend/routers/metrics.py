"""Router FastAPI fournissant les métriques d'évaluation et l'état de santé du système."""

from fastapi import APIRouter
from backend.services.model_service import load_metrics

router = APIRouter(prefix="/api", tags=["Métriques & Santé"])


@router.get(
    "/metrics",
    summary="Métriques de performance certifiées des trois agents",
    description="Retourne les ROC-AUC, scores de rappel et rapports de classification de chaque agent.",
)
def get_metrics():
    return load_metrics()


@router.get(
    "/health",
    summary="Vérification de l'état opérationnel du serveur",
    description="Retourne le statut d'activité de l'API de triage.",
)
def get_health():
    return {
        "status": "online",
        "service": "MedTriage AI Multi-Agent Backend",
        "version": "1.2.0",
    }
