"""Démonstration du système multi-agent de triage médical.

Exemple :
    python demo.py --image data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg \
                   --age 67 --gender Male --hypertension 1 --heart-disease 0 \
                   --ever-married Yes --work-type Private --residence Urban \
                   --glucose 228.7 --bmi 36.6 --smoking "formerly smoked"

Valeurs possibles :
    gender        : Male | Female | Other
    work-type     : Private | Self-employed | Govt_job | children | Never_worked
    residence     : Urban | Rural
    smoking       : formerly smoked | never smoked | smokes | Unknown

ATTENTION : projet pédagogique, ne constitue pas un outil de diagnostic médical.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent


def agent_image(image_path):
    """Agent 1 : probabilité de pneumonie à partir de la radiographie."""
    import tensorflow as tf  # import tardif : TensorFlow est lent à charger

    model = tf.keras.models.load_model(ROOT / "agent_image" / "model_cnn.keras")
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)[None, ...]
    return float(model.predict(arr, verbose=0)[0][0])


def agent_clinique(row):
    """Agent 2 : probabilité d'AVC à partir des données cliniques."""
    model = joblib.load(ROOT / "agent_clinique" / "model_rf.joblib")
    return float(model.predict_proba(pd.DataFrame([row]))[0, 1])


def agent_decision(p_img, p_clin):
    """Agent 3 : fusion des deux probabilités -> décision de triage."""
    model = joblib.load(ROOT / "agent_decision" / "model_lr.joblib")
    p = float(model.predict_proba([[p_img, p_clin]])[0, 1])
    if p >= 0.7:
        niveau = "URGENT"
    elif p >= 0.4:
        niveau = "À SURVEILLER"
    else:
        niveau = "NON URGENT"
    return p, niveau


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--image", required=True, help="chemin de la radiographie")
    ap.add_argument("--age", type=float, required=True)
    ap.add_argument("--gender", default="Male")
    ap.add_argument("--hypertension", type=int, default=0, choices=[0, 1])
    ap.add_argument("--heart-disease", type=int, default=0, choices=[0, 1])
    ap.add_argument("--ever-married", default="Yes")
    ap.add_argument("--work-type", default="Private")
    ap.add_argument("--residence", default="Urban")
    ap.add_argument("--glucose", type=float, required=True, help="glycémie moyenne")
    ap.add_argument("--bmi", type=float, default=None, help="IMC (vide si inconnu)")
    ap.add_argument("--smoking", default="never smoked")
    a = ap.parse_args()

    clinical = {
        "age": a.age,
        "avg_glucose_level": a.glucose,
        "bmi": a.bmi if a.bmi is not None else float("nan"),
        "hypertension": a.hypertension,
        "heart_disease": a.heart_disease,
        "gender": a.gender,
        "ever_married": a.ever_married,
        "work_type": a.work_type,
        "Residence_type": a.residence,
        "smoking_status": a.smoking,
    }

    p_img = agent_image(a.image)
    p_clin = agent_clinique(clinical)
    p_final, niveau = agent_decision(p_img, p_clin)

    print("=" * 48)
    print(" TRIAGE MÉDICAL — RÉSULTAT")
    print("=" * 48)
    print(f" Agent Image     (CNN)   : risque pneumonie = {p_img:.1%}")
    print(f" Agent Clinique  (RF)    : risque AVC       = {p_clin:.1%}")
    print(f" Agent Décision  (LogReg): score d'urgence  = {p_final:.1%}")
    print("-" * 48)
    print(f" DÉCISION : {niveau}")
    print("=" * 48)
    print(" Outil pédagogique, ne remplace pas un avis médical.")


if __name__ == "__main__":
    main()
