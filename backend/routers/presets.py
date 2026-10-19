"""Router FastAPI fournissant les profils cliniques préconfigurés et les cas aléatoires du dataset."""

import random
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np

from backend.config import PRESETS, STROKE_CSV, CHEST_XRAY_DIR

router = APIRouter(prefix="/api", tags=["Presets & Dataset Cas Aléatoires"])

# Cache en mémoire pour réponses instantanées (< 5ms)
_stroke_df: Optional[pd.DataFrame] = None
_normal_imgs = None
_pneu_imgs = None


def _get_dataset_resources():
    global _stroke_df, _normal_imgs, _pneu_imgs
    if _stroke_df is None and STROKE_CSV.exists():
        _stroke_df = pd.read_csv(STROKE_CSV)
        _stroke_df["bmi"] = pd.to_numeric(_stroke_df["bmi"], errors="coerce")

    if _normal_imgs is None and (CHEST_XRAY_DIR / "test" / "NORMAL").exists():
        _normal_imgs = list((CHEST_XRAY_DIR / "test" / "NORMAL").glob("*.jpeg")) + list((CHEST_XRAY_DIR / "test" / "NORMAL").glob("*.jpg"))
        _pneu_imgs = list((CHEST_XRAY_DIR / "test" / "PNEUMONIA").glob("*.jpeg")) + list((CHEST_XRAY_DIR / "test" / "PNEUMONIA").glob("*.jpg"))

    return _stroke_df, _normal_imgs, _pneu_imgs


@router.get(
    "/presets",
    summary="Récupération des cas cliniques préconfigurés",
    description="Retourne la liste des profils types (Pneumonie, AVC, Cas Mixte, Ambulatoire) pour tests et démo.",
)
def get_presets():
    return PRESETS


@router.get(
    "/random-case",
    summary="Tirage d'un patient et d'une radiographie aléatoires du dataset réel",
    description="Sélectionne un profil patient réel dans le Stroke Dataset et une radio réelle dans Chest X-Ray.",
)
def get_random_case(quadrant: str = Query("any", description="Cibler un quadrant: any, Q00, Q10, Q01, Q11")):
    df, normals, pneus = _get_dataset_resources()
    if df is None or not normals or not pneus:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Les fichiers du dataset local (data/) ne sont pas disponibles sur le serveur."
        )

    # Filtrage selon le quadrant pivot désiré
    if quadrant in ("Q01", "Q11"):
        sub_df = df[df["stroke"] == 1]
    elif quadrant in ("Q00", "Q10"):
        sub_df = df[df["stroke"] == 0]
    else:
        sub_df = df

    row = sub_df.sample(1).iloc[0]

    # Sélection de la radiographie
    if quadrant in ("Q10", "Q11"):
        img = random.choice(pneus)
        img_label = "PNEUMONIA"
    elif quadrant in ("Q00", "Q01"):
        img = random.choice(normals)
        img_label = "NORMAL"
    else:
        if random.random() < 0.5:
            img = random.choice(normals)
            img_label = "NORMAL"
        else:
            img = random.choice(pneus)
            img_label = "PNEUMONIA"

    rel_path = f"test/{img_label}/{img.name}"
    stroke_val = int(row["stroke"])
    bmi_val = round(float(row["bmi"]), 1) if not pd.isna(row["bmi"]) else 24.5

    return {
        "source": "Kaggle Datasets (data/)",
        "patient_id": int(row["id"]),
        "age": float(row["age"]),
        "gender": str(row["gender"]) if str(row["gender"]) in ("Male", "Female", "Other") else "Male",
        "glucose": round(float(row["avg_glucose_level"]), 1),
        "bmi": bmi_val,
        "hypertension": int(row["hypertension"]),
        "heart_disease": int(row["heart_disease"]),
        "ever_married": str(row["ever_married"]) if str(row["ever_married"]) in ("Yes", "No") else "Yes",
        "work_type": str(row["work_type"]) if str(row["work_type"]) in ("Private", "Self-employed", "Govt_job", "children", "Never_worked") else "Private",
        "residence": str(row["Residence_type"]) if str(row["Residence_type"]) in ("Urban", "Rural") else "Urban",
        "smoking": str(row["smoking_status"]) if str(row["smoking_status"]) in ("formerly smoked", "never smoked", "smokes", "Unknown") else "never smoked",
        "true_stroke": stroke_val,
        "stroke_label": "AVC Avéré (stroke=1)" if stroke_val == 1 else "Pas d'AVC (stroke=0)",
        "image_filename": img.name,
        "image_class": img_label,
        "sample_token": f"dataset:{rel_path}",
        "image_url": f"/api/dataset-image?path={rel_path}",
        "ground_truth_summary": f"Patient #{int(row['id'])} ({'AVC Avéré' if stroke_val==1 else 'Pas d AVC'}) · Radio {img_label} ({img.name})",
    }


@router.get(
    "/dataset-image",
    summary="Récupération du flux image d'une radio du dataset",
    description="Sert le fichier radiologique JPEG directement depuis le dossier data/chest_xray/.",
)
def get_dataset_image(path: str = Query(..., description="Chemin relatif dans chest_xray")):
    rel = path.replace("\\", "/").strip("/")
    full_path = (CHEST_XRAY_DIR / rel).resolve()
    if not full_path.exists() or not str(full_path).startswith(str(CHEST_XRAY_DIR.resolve())):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image du dataset '{rel}' introuvable ou chemin non autorisé."
        )
    return FileResponse(full_path, media_type="image/jpeg")
