"""Service de chargement, de mise en cache et de réchauffage dynamique des modèles d'IA."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np

from backend.config import METRICS_PATHS, MODEL_PATHS

# Cache mémoire des modèles et des métriques avec suivi des horodatages de modification
MODELS = {
    "cnn": None,
    "rf": None,
    "lr": None,
}
MODELS_MTIMES: Dict[str, float] = {}

METRICS_CACHE: Dict[str, dict] = {}
METRICS_MTIMES: Dict[str, float] = {}


def extract_agent_metrics(raw: dict, agent_key: str) -> dict:
    """Extrait de façon standardisée les métriques clés (roc_auc, accuracy, recall, precision, f1).
    
    Supporte de manière robuste les variantes de clés et structures des différents rapports scikit-learn.
    """
    if not raw or not isinstance(raw, dict):
        return {
            "roc_auc": 0.0,
            "accuracy": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
        }

    roc_auc = float(raw.get("roc_auc", 0.0))
    rep = raw.get("report", {})

    # 1. Accuracy globale
    accuracy = 0.0
    if "accuracy" in rep and isinstance(rep["accuracy"], (int, float)):
        accuracy = float(rep["accuracy"])
    elif "weighted avg" in rep and isinstance(rep["weighted avg"], dict) and "recall" in rep["weighted avg"]:
        accuracy = float(rep["weighted avg"]["recall"])

    # 2. Classe positive / cible selon l'agent
    target_class_keys = []
    if agent_key in ("image", "cnn"):
        target_class_keys = ["PNEUMONIA", "pneumonia", "1", 1]
    elif agent_key in ("clinique", "rf"):
        target_class_keys = ["1", 1, "AVC", "stroke"]
    elif agent_key in ("decision", "lr"):
        target_class_keys = ["Urgent", "urgent", "1", 1]
    else:
        target_class_keys = ["Urgent", "PNEUMONIA", "1", 1]

    class_dict = None
    for k in target_class_keys:
        if k in rep and isinstance(rep[k], dict):
            class_dict = rep[k]
            break

    if not class_dict:
        # Fallback première classe non agrégée
        for k, v in rep.items():
            if isinstance(v, dict) and k not in ("accuracy", "macro avg", "weighted avg") and ("recall" in v or "precision" in v):
                class_dict = v
                break

    if not class_dict and "macro avg" in rep and isinstance(rep["macro avg"], dict):
        class_dict = rep["macro avg"]

    precision = float(class_dict.get("precision", 0.0)) if class_dict else 0.0
    recall = float(class_dict.get("recall", 0.0)) if class_dict else 0.0
    f1 = float(class_dict.get("f1-score", class_dict.get("f1", 0.0))) if class_dict else 0.0

    return {
        "roc_auc": round(roc_auc, 4),
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


def load_metrics(force_reload: bool = False) -> Dict[str, Any]:
    """Charge et synchronise dynamiquement les métriques des trois agents.
    
    Vérifie les timestamps (st_mtime) des fichiers JSON sur disque : si un fichier change
    (ex: réentraînement d'Agent 3 ou modification des seuils), il est immédiatement rechargé
    sans nécessiter de redémarrage de serveur ni de modification de code.
    """
    for key, path in METRICS_PATHS.items():
        if not path.exists():
            if key not in METRICS_CACHE:
                METRICS_CACHE[key] = {}
            continue

        try:
            current_mtime = path.stat().st_mtime
            if force_reload or key not in METRICS_CACHE or METRICS_MTIMES.get(key) != current_mtime:
                METRICS_CACHE[key] = json.loads(path.read_text(encoding="utf-8"))
                METRICS_MTIMES[key] = current_mtime
        except Exception:
            if key not in METRICS_CACHE:
                METRICS_CACHE[key] = {}

    # Extraction des métriques pour chaque agent
    img_met = extract_agent_metrics(METRICS_CACHE.get("image", {}), "image")
    clin_met = extract_agent_metrics(METRICS_CACHE.get("clinique", {}), "clinique")
    dec_met = extract_agent_metrics(METRICS_CACHE.get("decision", {}), "decision")

    coefs = METRICS_CACHE.get("decision", {}).get("coefficients", {})
    coef_img = float(coefs.get("image", 2.99))
    coef_clin = float(coefs.get("clinique", 4.24))
    intercept = float(coefs.get("intercept", -3.31))
    sign = "+" if intercept >= 0 else "-"
    formula_str = f"z = ({coef_img:.2f} × P_img) + ({coef_clin:.2f} × P_clin) {sign} {abs(intercept):.2f}"

    METRICS_CACHE["summary"] = {
        "roc_auc": {
            "cnn": img_met["roc_auc"],
            "rf": clin_met["roc_auc"],
            "lr": dec_met["roc_auc"],
        },
        "accuracy": {
            "cnn": img_met["accuracy"],
            "rf": clin_met["accuracy"],
            "lr": dec_met["accuracy"],
        },
        "recall": {
            "cnn": img_met["recall"],
            "rf": clin_met["recall"],
            "lr": dec_met["recall"],
        },
        "precision": {
            "cnn": img_met["precision"],
            "rf": clin_met["precision"],
            "lr": dec_met["precision"],
        },
        "f1": {
            "cnn": img_met["f1"],
            "rf": clin_met["f1"],
            "lr": dec_met["f1"],
        },
    }
    METRICS_CACHE["decision_formula"] = formula_str
    METRICS_CACHE["coefficients"] = {
        "image": coef_img,
        "clinique": coef_clin,
        "intercept": intercept,
    }

    return METRICS_CACHE


def get_models() -> Tuple[object, object, object]:
    """Charge paresseusement les 3 modèles en RAM et recharge si un fichier a été modifié."""
    # 1. Modèle CNN MobileNetV2 (Agent 1 Vision)
    cnn_path = MODEL_PATHS["cnn"]
    if cnn_path.exists():
        mtime = cnn_path.stat().st_mtime
        if MODELS["cnn"] is None or MODELS_MTIMES.get("cnn") != mtime:
            import tensorflow as tf
            MODELS["cnn"] = tf.keras.models.load_model(cnn_path)
            MODELS_MTIMES["cnn"] = mtime

    # 2. Modèle Random Forest (Agent 2 Clinique)
    rf_path = MODEL_PATHS["rf"]
    if rf_path.exists():
        mtime = rf_path.stat().st_mtime
        if MODELS["rf"] is None or MODELS_MTIMES.get("rf") != mtime:
            MODELS["rf"] = joblib.load(rf_path)
            MODELS_MTIMES["rf"] = mtime

    # 3. Modèle Régression Logistique (Agent 3 Consensus — surcharge optionnelle via MEDTRIAGE_LR_PATH)
    env_lr = os.environ.get("MEDTRIAGE_LR_PATH")
    lr_path = Path(env_lr) if env_lr else MODEL_PATHS["lr"]
    if lr_path.exists():
        mtime = (str(lr_path), lr_path.stat().st_mtime)
        if MODELS["lr"] is None or MODELS_MTIMES.get("lr") != mtime:
            loaded_lr = joblib.load(lr_path)
            if not hasattr(loaded_lr, "multi_class"):
                setattr(loaded_lr, "multi_class", "auto")
            MODELS["lr"] = loaded_lr
            MODELS_MTIMES["lr"] = mtime

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
