"""Router FastAPI pour consulter et gérer l'historique des triages sauvegardés."""

from fastapi import APIRouter, Query
from backend.services.history_service import get_history, clear_history

router = APIRouter(prefix="/api", tags=["Historique des Triages"])


@router.get(
    "/history",
    summary="Récupérer l'historique des évaluations sauvegardées",
    description=(
        "Retourne les N dernières évaluations de triage enregistrées dans le fichier "
        "data/historique_triages.json. Les résultats sont triés du plus récent au plus ancien."
    ),
)
def get_triage_history(limit: int = Query(default=20, ge=1, le=100)):
    records = get_history(limit=limit)
    return {
        "total": len(records),
        "evaluations": records,
    }


@router.delete(
    "/history",
    summary="Effacer tout l'historique des triages",
    description="Supprime toutes les évaluations enregistrées dans le fichier JSON.",
)
def delete_triage_history():
    count = clear_history()
    return {
        "message": f"{count} évaluation(s) supprimée(s) avec succès.",
        "total_restant": 0,
    }
