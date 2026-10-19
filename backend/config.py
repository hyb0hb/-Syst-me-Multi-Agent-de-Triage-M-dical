"""Configuration centrale du backend MedTriage AI."""

from pathlib import Path

# Chemins principaux
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
STATIC_DIR = ROOT_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "samples"
DATA_DIR = ROOT_DIR / "data"
STROKE_CSV = DATA_DIR / "healthcare-dataset-stroke-data.csv"
CHEST_XRAY_DIR = DATA_DIR / "chest_xray"

# Chemins des modèles d'agents
MODEL_PATHS = {
    "cnn": ROOT_DIR / "agent_image" / "model_cnn.keras",
    "rf": ROOT_DIR / "agent_clinique" / "model_rf.joblib",
    "lr": ROOT_DIR / "agent_decision" / "model_lr.joblib",
}

# Chemins des fichiers de métriques certifiées
METRICS_PATHS = {
    "image": ROOT_DIR / "agent_image" / "metrics.json",
    "clinique": ROOT_DIR / "agent_clinique" / "metrics.json",
    "decision": ROOT_DIR / "agent_decision" / "metrics.json",
}

# Profils cliniques types (Presets de démonstration)
PRESETS = {
    "cas_pneumonie": {
        "title": "Cas 1 : Détresse Respiratoire Aiguë",
        "description": "Patient 54 ans avec fièvre, toux et opacité alvéolaire suspecte.",
        "sample_image": "/static/samples/pneumonia.jpg",
        "age": 54,
        "gender": "Male",
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "residence": "Urban",
        "glucose": 98.4,
        "bmi": 26.2,
        "smoking": "never smoked",
    },
    "cas_avc": {
        "title": "Cas 2 : Suspicion AVC Ischémique",
        "description": "Patiente 72 ans, antécédents d'hypertension et hyperglycémie.",
        "sample_image": "/static/samples/normal.jpg",
        "age": 72,
        "gender": "Female",
        "hypertension": 1,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Self-employed",
        "residence": "Rural",
        "glucose": 215.3,
        "bmi": 34.8,
        "smoking": "formerly smoked",
    },
    "cas_critique_mixte": {
        "title": "Cas 3 : Urgence Vitale Mixte (Choc & Comorbidités)",
        "description": "Patient 79 ans, foyer infectieux pulmonaire et risque cardiovasculaire élevé.",
        "sample_image": "/static/samples/pneumonia.jpg",
        "age": 79,
        "gender": "Male",
        "hypertension": 1,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "residence": "Urban",
        "glucose": 242.0,
        "bmi": 31.5,
        "smoking": "smokes",
    },
    "cas_routine": {
        "title": "Cas 4 : Contrôle de Routine (Bilan Ambulatoire)",
        "description": "Patient 28 ans en bonne santé sans signe de gravité.",
        "sample_image": "/static/samples/normal.jpg",
        "age": 28,
        "gender": "Female",
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "No",
        "work_type": "Private",
        "residence": "Urban",
        "glucose": 86.5,
        "bmi": 21.8,
        "smoking": "never smoked",
    },
}
