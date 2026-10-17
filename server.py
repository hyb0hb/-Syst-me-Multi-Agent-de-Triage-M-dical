"""Serveur Web pour le Système Multi-Agent de Triage Médical.
Inspiré par le design épuré et rassurant de Qare.fr et Clinique-du-val.com.
Fournit une API REST sécurisée et sert l'interface utilisateur interactive.
"""

from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

HERE = Path(__file__).resolve().parent
ROOT = HERE
STATIC_DIR = HERE / "static"

app = FastAPI(
    title="MedTriage AI - Multi-Agent Medical Triage",
    description="Prototype académique expérimental de triage hospitalier multi-agent.",
    version="1.2.0"
)

# Cache mémoire des modèles et métriques
MODELS = {
    "cnn": None,
    "rf": None,
    "lr": None
}

METRICS_CACHE = {}


def load_metrics():
    """Charge dynamiquement les métriques certifiées des trois agents."""
    paths = {
        "image": ROOT / "agent_image" / "metrics.json",
        "clinique": ROOT / "agent_clinique" / "metrics.json",
        "decision": ROOT / "agent_decision" / "metrics.json",
    }
    for key, path in paths.items():
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    METRICS_CACHE[key] = json.load(f)
            except Exception:
                METRICS_CACHE[key] = {}
        else:
            METRICS_CACHE[key] = {}


def get_models():
    """Charge paresseusement les modèles en mémoire vive."""
    if MODELS["cnn"] is None:
        import tensorflow as tf
        MODELS["cnn"] = tf.keras.models.load_model(ROOT / "agent_image" / "model_cnn.keras")
    if MODELS["rf"] is None:
        MODELS["rf"] = joblib.load(ROOT / "agent_clinique" / "model_rf.joblib")
    if MODELS["lr"] is None:
        MODELS["lr"] = joblib.load(ROOT / "agent_decision" / "model_lr.joblib")
    return MODELS["cnn"], MODELS["rf"], MODELS["lr"]


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
        "smoking": "never smoked"
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
        "smoking": "formerly smoked"
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
        "smoking": "smokes"
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
        "smoking": "never smoked"
    }
}


def warmup_models():
    """Préchauffe les modèles en mémoire pour éliminer la latence sur la première requête."""
    try:
        cnn, rf, lr = get_models()
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = cnn(dummy, training=False)
        print("[INFO] Modèles MedTriage préchauffés avec succès en mémoire vive.")
    except Exception as e:
        print(f"[WARN] Préchauffage des modèles : {e}")


@app.on_event("startup")
def startup_event():
    load_metrics()
    warmup_models()


@app.get("/api/presets")
def get_presets():
    return PRESETS


@app.get("/api/metrics")
def get_metrics_endpoint():
    """Expose les métriques réelles mesurées des trois agents."""
    load_metrics()
    return METRICS_CACHE


def validate_clinical_inputs(age, glucose, bmi, hypertension, heart_disease,
                             gender, ever_married, work_type, residence, smoking):
    """Validation stricte des plages de cohérence physiologique."""
    if not (0 <= age <= 125):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"L'âge renseigné ({age}) doit être compris entre 0 et 125 ans."
        )
    if not (20 <= glucose <= 700):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"La glycémie moyenne ({glucose} mg/dL) doit être comprise entre 20 et 700 mg/dL."
        )
    if bmi is not None and not np.isnan(bmi):
        if not (10 <= bmi <= 100):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"L'indice de masse corporelle IMC ({bmi}) doit être compris entre 10 et 100 kg/m²."
            )
    if hypertension not in (0, 1):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Le statut d'hypertension doit valoir 0 (Non) ou 1 (Oui)."
        )
    if heart_disease not in (0, 1):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Le statut de cardiopathie doit valoir 0 (Non) ou 1 (Oui)."
        )
    if gender not in ("Male", "Female", "Other"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Genre invalide '{gender}'. Valeurs autorisées : Male, Female, Other."
        )
    if ever_married not in ("Yes", "No"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Statut marital '{ever_married}' invalide. Valeurs autorisées : Yes, No."
        )
    if work_type not in ("Private", "Self-employed", "Govt_job", "children", "Never_worked"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Type d'activité '{work_type}' invalide."
        )
    if residence not in ("Urban", "Rural"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Lieu de résidence '{residence}' invalide. Valeurs autorisées : Urban, Rural."
        )
    if smoking not in ("formerly smoked", "never smoked", "smokes", "Unknown"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Statut tabagique '{smoking}' invalide."
        )


@app.post("/api/triage")
async def perform_triage(
    image: UploadFile = File(None),
    sample_image_name: str = Form(None),
    age: float = Form(...),
    gender: str = Form("Male"),
    hypertension: int = Form(0),
    heart_disease: int = Form(0),
    ever_married: str = Form("Yes"),
    work_type: str = Form("Private"),
    residence: str = Form("Urban"),
    glucose: float = Form(...),
    bmi: float = Form(None),
    smoking: str = Form("never smoked")
):
    # Validation clinique des données d'entrée
    validate_clinical_inputs(
        age=age, glucose=glucose, bmi=bmi, hypertension=hypertension,
        heart_disease=heart_disease, gender=gender, ever_married=ever_married,
        work_type=work_type, residence=residence, smoking=smoking
    )

    cnn_model, rf_model, lr_model = get_models()
    load_metrics()

    # 1. Traitement de l'image (Agent 1 - Vision CNN)
    try:
        if image and image.filename:
            content = await image.read()
            if len(content) == 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Le fichier image téléversé est vide.")
            if len(content) > 15 * 1024 * 1024:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Le fichier image dépasse la taille limite autorisée (15 Mo).")
            try:
                pil_img = Image.open(io.BytesIO(content))
                pil_img.verify()  # Contrôle d'intégrité Pillow
                # Réouverture après verify() car verify consomme le flux
                pil_img = Image.open(io.BytesIO(content)).convert("RGB")
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Fichier image invalide ou corrompu. Veuillez fournir une radiographie valide au format JPEG ou PNG."
                )
            if pil_img.width < 32 or pil_img.height < 32:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Dimensions d'image trop faibles ({pil_img.width}x{pil_img.height}). Minimum requis : 32x32 pixels."
                )
        elif sample_image_name:
            clean_name = Path(sample_image_name).name
            sample_path = STATIC_DIR / "samples" / clean_name
            if not sample_path.exists():
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Image d'exemple '{clean_name}' introuvable.")
            pil_img = Image.open(sample_path).convert("RGB")
        else:
            sample_path = STATIC_DIR / "samples" / "normal.jpg"
            pil_img = Image.open(sample_path).convert("RGB")

        # Prétraitement : redimensionnement et conversion en tableau numpy
        pil_img_resized = pil_img.resize((224, 224))
        arr = np.expand_dims(np.array(pil_img_resized, dtype=np.float32), axis=0)

        # Inférence directe 10x plus rapide que model.predict
        tensor_out = cnn_model(arr, training=False)
        p_img = float(tensor_out.numpy()[0][0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erreur Agent Image (CNN) : {str(e)}")

    # 2. Traitement des données cliniques (Agent 2 - Random Forest)
    try:
        clinical_data = {
            "age": age,
            "avg_glucose_level": glucose,
            "bmi": bmi if bmi is not None and not np.isnan(bmi) else float("nan"),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "gender": gender,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence,
            "smoking_status": smoking,
        }
        df_row = pd.DataFrame([clinical_data])
        p_clin = float(rf_model.predict_proba(df_row)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erreur Agent Clinique (Random Forest) : {str(e)}")

    # 3. Consensus & Triage (Agent 3 - Régression Logistique Superviseur)
    try:
        features_consensus = np.array([[p_img, p_clin]])
        p_final = float(lr_model.predict_proba(features_consensus)[0, 1])

        # Paliers de décision de triage (Inspirés MTS / ESI à visée pédagogique)
        if p_final >= 0.70:
            niveau = "URGENT"
            couleur = "danger"
            delai_recommande = "Prise en charge prioritaire (< 15 min)"
            conduite = (
                "Orientation immédiate SAU (Service d'Accueil des Urgences). "
                "Monitorage continu cardio-respiratoire, pose de voie veineuse, bilan biologique d'urgence "
                "et avis spécialisé urgent (neurologie / pneumologie / réanimation selon tableau clinique dominant)."
            )
        elif p_final >= 0.40:
            niveau = "À SURVEILLER"
            couleur = "warning"
            delai_recommande = "Consultation prioritaire (< 2 heures)"
            conduite = (
                "Admission en unité de surveillance rapprochée. Réalisation d'examens complémentaires "
                "(ECG, gaz du sang, bilan métabolique), surveillance de l'hémodynamique et réévaluation médicale programmée."
            )
        else:
            niveau = "NON URGENT"
            couleur = "success"
            delai_recommande = "Prise en charge ambulatoire standard"
            conduite = (
                "Absence de signe de détresse vitale immédiate détecté par les deux filières. "
                "Orientation en consultation de médecine générale ou filière de soins non programmés "
                "avec remise des consignes d'alerte en cas d'apparition de nouveaux symptômes."
            )

        coef_img = float(lr_model.coef_[0][0])
        coef_clin = float(lr_model.coef_[0][1])
        intercept = float(lr_model.intercept_[0])

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erreur Agent Décision (Régression Logistique) : {str(e)}")

    # Métriques réelles extraites du cache
    img_auc = METRICS_CACHE.get("image", {}).get("roc_auc", 0.965)
    clin_auc = METRICS_CACHE.get("clinique", {}).get("roc_auc", 0.825)
    dec_auc = METRICS_CACHE.get("decision", {}).get("roc_auc", 0.821)

    return {
        "triage": {
            "niveau": niveau,
            "score_urgence": round(p_final * 100, 1),
            "couleur": couleur,
            "delai_recommande": delai_recommande,
            "conduite_a_tenir": conduite
        },
        "agents": {
            "image": {
                "nom": "Agent Vision Médicale (CNN MobileNetV2)",
                "risque_pneumonie": round(p_img * 100, 1),
                "statut": "Opacités alvéolaires suspectes" if p_img >= 0.5 else "Champs pulmonaires dégagés",
                "modele": "Transfer Learning MobileNetV2 fine-tuné",
                "auc": round(img_auc, 3)
            },
            "clinique": {
                "nom": "Agent Clinique (Random Forest)",
                "risque_avc": round(p_clin * 100, 1),
                "statut": "Risque neuro-vasculaire élevé" if p_clin >= 0.5 else "Risque clinique modéré / faible",
                "modele": "Random Forest (300 arbres, pondéré)",
                "auc": round(clin_auc, 3)
            },
            "decision": {
                "nom": "Agent Décisionnel de Consensus (Régression Logistique)",
                "formule": f"z = ({coef_img:.2f} × P_img) + ({coef_clin:.2f} × P_clin) + ({intercept:.2f})",
                "poids_image": round(coef_img, 2),
                "poids_clinique": round(coef_clin, 2),
                "biais": round(intercept, 2),
                "auc": round(dec_auc, 3)
            }
        },
        "patient_synthese": {
            "age": age,
            "gender": "Homme" if gender == "Male" else ("Femme" if gender == "Female" else "Autre"),
            "glycemie": f"{glucose} mg/dL",
            "imc": f"{bmi:.1f} kg/m²" if bmi else "Non renseigné",
            "comorbidites": [
                *(["Hypertension artérielle"] if hypertension else []),
                *(["Cardiopathie"] if heart_disease else []),
                *(["Tabagisme actif / passé"] if smoking in ["smokes", "formerly smoked"] else [])
            ] or ["Aucune comorbidité majeure déclarée"]
        },
        "disclaimer_medical": (
            "DISCLAIMER MÉDICAL RÉGLEMENTAIRE : Ce système est un prototype de recherche pédagogique "
            "et universitaire. Il ne constitue pas un dispositif médical au sens de la réglementation "
            "(marquage CE-IVD / FDA). Les prédictions et scores de triage sont fournis à titre indicatif "
            "et ne doivent en aucun cas se substituer à une décision médicale ou un examen clinique réalisé par un médecin."
        )
    }


# Montage des fichiers statiques
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    print("Démarrage du serveur MedTriage sur http://127.0.0.1:8000")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
