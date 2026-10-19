"""Service de chargement, de mise en cache et de réchauffage des modèles d'IA."""

import json
from typing import Dict, Tuple
import joblib
import numpy as np

from backend.config import MODEL_PATHS, METRICS_PATHS

# Cache mémoire pour éviter de recharger les poids sur chaque requête
MODELS = {
    "cnn": None,
    "rf": None,
    "lr": None,
}

METRICS_CACHE: Dict[str, dict] = {}


def load_metrics() -> Dict[str, dict]:
    """Charge les métriques certifiées des trois agents (une seule fois en cache)."""
    if METRICS_CACHE:  # Déjà chargé — pas de relecture disque inutile
        return METRICS_CACHE
    for key, path in METRICS_PATHS.items():
        try:
            METRICS_CACHE[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            METRICS_CACHE[key] = {}
    return METRICS_CACHE


def get_models() -> Tuple[object, object, object]:
    """Charge paresseusement les 3 modèles en mémoire vive (RAM)."""
    if MODELS["cnn"] is None:
        import tensorflow as tf

        MODELS["cnn"] = tf.keras.models.load_model(MODEL_PATHS["cnn"])
    if MODELS["rf"] is None:
        MODELS["rf"] = joblib.load(MODEL_PATHS["rf"])
    if MODELS["lr"] is None:
        MODELS["lr"] = joblib.load(MODEL_PATHS["lr"])
    return MODELS["cnn"], MODELS["rf"], MODELS["lr"]


def warmup_models() -> None:
    """Préchauffe les modèles pour éliminer la latence sur la première requête."""
    try:
        cnn, rf, lr = get_models()
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = cnn(dummy, training=False)
        print("[INFO] Modèles MedTriage préchauffés avec succès en mémoire vive.")
    except Exception as e:
        print(f"[WARN] Préchauffage des modèles : {e}")
