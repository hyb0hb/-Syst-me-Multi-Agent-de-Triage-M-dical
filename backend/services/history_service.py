"""Service de persistance des évaluations de triage dans un fichier JSON.

Chaque triage exécuté est automatiquement sauvegardé dans :
  data/historique_triages.json

Le fichier garde les 100 dernières évaluations (FIFO).
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

# Fichier de stockage des évaluations
HISTORY_FILE = Path(__file__).resolve().parents[2] / "data" / "historique_triages.json"
MAX_RECORDS = 100  # On garde les 100 derniers triages maximum


def _load_history() -> list:
    """Charge l'historique depuis le fichier JSON. Retourne [] si inexistant."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(records: list) -> None:
    """Écrit la liste des évaluations dans le fichier JSON."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def save_triage_result(result: dict, patient_info: dict) -> str:
    """Enregistre un résultat de triage dans l'historique JSON.

    Args:
        result: Le dictionnaire complet retourné par run_full_triage().
        patient_info: Métadonnées brutes du patient (âge, genre...).

    Returns:
        L'identifiant unique de l'évaluation sauvegardée (ex: EVAL-a3f2...).
    """
    records = _load_history()

    eval_id = "EVAL-" + str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().isoformat()

    record = {
        "id": eval_id,
        "timestamp": timestamp,
        "date_heure": datetime.now().strftime("%d/%m/%Y à %H:%M"),
        # Résultat du triage
        "niveau": result["triage"]["niveau"],
        "score_urgence": result["triage"]["score_urgence"],
        "couleur": result["triage"]["couleur"],
        "delai_recommande": result["triage"]["delai_recommande"],
        "conduite_a_tenir": result["triage"]["conduite_a_tenir"],
        # Scores des agents
        "agent_image_risque": result["agents"]["image"]["risque_pneumonie"],
        "agent_clinique_risque": result["agents"]["clinique"]["risque_avc"],
        # Synthèse patient
        "patient": result["patient_synthese"],
        # Info brutes
        "parametres_bruts": patient_info,
    }

    # Insertion en tête de liste (plus récent en premier)
    records.insert(0, record)

    # On ne garde que les MAX_RECORDS derniers
    if len(records) > MAX_RECORDS:
        records = records[:MAX_RECORDS]

    _save_history(records)
    return eval_id


def get_history(limit: int = 20) -> list:
    """Retourne les N évaluations les plus récentes.

    Args:
        limit: Nombre maximum d'entrées à retourner (défaut 20).

    Returns:
        Liste des évaluations triées du plus récent au plus ancien.
    """
    records = _load_history()
    return records[:limit]


def clear_history() -> int:
    """Efface tout l'historique. Retourne le nombre d'entrées supprimées."""
    records = _load_history()
    count = len(records)
    _save_history([])
    return count
