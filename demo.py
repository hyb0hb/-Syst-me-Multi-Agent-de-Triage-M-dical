"""Démonstration en ligne de commande du Système Multi-Agent de Triage Médical.

Exemples d'utilisation :
------------------------
1. Avec une image d'exemple intégrée :
    py demo.py --sample pneumonia --age 54 --gender Male --hypertension 0 --heart-disease 0 \
               --glucose 98.4 --bmi 26.2 --smoking "never smoked"

2. Avec un fichier image local :
    py demo.py --image static/samples/normal.jpg --age 72 --gender Female --hypertension 1 \
               --heart-disease 1 --glucose 215.3 --bmi 34.8 --smoking "formerly smoked"

Valeurs admises :
    gender        : Male | Female | Other
    work-type     : Private | Self-employed | Govt_job | children | Never_worked
    residence     : Urban | Rural
    smoking       : formerly smoked | never smoked | smokes | Unknown

AVERTISSEMENT : Prototype universitaire expérimental — ne constitue pas un avis médical.
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent


def agent_image(image_path):
    """Agent 1 : probabilité de pneumonie à partir du cliché radiographique."""
    import tensorflow as tf

    model_path = ROOT / "agent_image" / "model_cnn.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle CNN introuvable : {model_path}")

    model = tf.keras.models.load_model(model_path)
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)[None, ...]
    return float(model.predict(arr, verbose=0)[0][0])


def agent_clinique(row):
    """Agent 2 : probabilité d'AVC à partir des données cliniques tabulaires."""
    model_path = ROOT / "agent_clinique" / "model_rf.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle Random Forest introuvable : {model_path}")

    model = joblib.load(model_path)
    return float(model.predict_proba(pd.DataFrame([row]))[0, 1])


def agent_decision(p_img, p_clin):
    """Agent 3 : superviseur de consensus (Régression Logistique)."""
    model_path = ROOT / "agent_decision" / "model_lr.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle Régression Logistique introuvable : {model_path}")

    model = joblib.load(model_path)
    p = float(model.predict_proba([[p_img, p_clin]])[0, 1])
    if p >= 0.70:
        niveau = "URGENT (Priorité SAU < 15 min)"
    elif p >= 0.40:
        niveau = "À SURVEILLER (Surveillance rapprochée < 2h)"
    else:
        niveau = "NON URGENT (Filière ambulatoire standard)"

    coef_img = float(model.coef_[0][0])
    coef_clin = float(model.coef_[0][1])
    intercept = float(model.intercept_[0])
    return p, niveau, coef_img, coef_clin, intercept


def validate_inputs(args, image_path):
    """Validation stricte des paramètres cliniques et de l'image."""
    # 1. Vérification de l'image
    if not image_path.exists():
        raise ValueError(f"Fichier image introuvable : {image_path}")
    try:
        with Image.open(image_path) as im:
            im.verify()
        with Image.open(image_path) as im:
            if im.width < 32 or im.height < 32:
                raise ValueError(f"Image trop petite ({im.width}x{im.height}px). Minimum 32x32 requis.")
    except Exception as e:
        raise ValueError(f"Fichier image invalide ou illisible : {e}")

    # 2. Bornes physiologiques
    if not (0 <= args.age <= 125):
        raise ValueError(f"Âge hors limites ({args.age} ans). Doit être entre 0 et 125.")
    if not (20 <= args.glucose <= 700):
        raise ValueError(f"Glycémie hors limites ({args.glucose} mg/dL). Doit être entre 20 et 700.")
    if args.bmi is not None and not (10 <= args.bmi <= 100):
        raise ValueError(f"IMC hors limites ({args.bmi} kg/m²). Doit être entre 10 et 100.")
    if args.gender not in ("Male", "Female", "Other"):
        raise ValueError(f"Genre invalide '{args.gender}'. Choix : Male, Female, Other.")
    if args.work_type not in ("Private", "Self-employed", "Govt_job", "children", "Never_worked"):
        raise ValueError(f"Type d'emploi invalide '{args.work_type}'.")
    if args.residence not in ("Urban", "Rural"):
        raise ValueError(f"Résidence invalide '{args.residence}'.")
    if args.smoking not in ("formerly smoked", "never smoked", "smokes", "Unknown"):
        raise ValueError(f"Statut tabagique invalide '{args.smoking}'.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--image", default=None, help="Chemin du cliché radiographique (JPEG/PNG)")
    ap.add_argument("--sample", default=None, choices=["normal", "pneumonia"],
                    help="Utiliser un échantillon prêt à l'emploi (normal ou pneumonia)")
    ap.add_argument("--age", type=float, required=True, help="Âge du patient (0-125)")
    ap.add_argument("--gender", default="Male", choices=["Male", "Female", "Other"])
    ap.add_argument("--hypertension", type=int, default=0, choices=[0, 1])
    ap.add_argument("--heart-disease", type=int, default=0, choices=[0, 1])
    ap.add_argument("--ever-married", default="Yes", choices=["Yes", "No"])
    ap.add_argument("--work-type", default="Private",
                    choices=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    ap.add_argument("--residence", default="Urban", choices=["Urban", "Rural"])
    ap.add_argument("--glucose", type=float, required=True, help="Glycémie moyenne en mg/dL (20-700)")
    ap.add_argument("--bmi", type=float, default=None, help="Indice de masse corporelle (10-100)")
    ap.add_argument("--smoking", default="never smoked",
                    choices=["formerly smoked", "never smoked", "smokes", "Unknown"])
    args = ap.parse_args()

    # Détermination de l'image
    if args.sample:
        image_path = ROOT / "static" / "samples" / f"{args.sample}.jpg"
    elif args.image:
        image_path = Path(args.image)
    else:
        image_path = ROOT / "static" / "samples" / "normal.jpg"
        print("[INFO] Aucune image spécifiée : utilisation par défaut de 'static/samples/normal.jpg'.")

    try:
        validate_inputs(args, image_path)
    except ValueError as err:
        print(f"\n[ERREUR DE VALIDATION] : {err}", file=sys.stderr)
        sys.exit(1)

    clinical = {
        "age": args.age,
        "avg_glucose_level": args.glucose,
        "bmi": args.bmi if args.bmi is not None else float("nan"),
        "hypertension": args.hypertension,
        "heart_disease": args.heart_disease,
        "gender": args.gender,
        "ever_married": args.ever_married,
        "work_type": args.work_type,
        "Residence_type": args.residence,
        "smoking_status": args.smoking,
    }

    print("\n" + "=" * 60)
    print(" SYSTÈME MULTI-AGENT DE TRIAGE MÉDICAL — DÉLIBÉRATION")
    print("=" * 60)
    print(f" Patient : {args.gender}, {args.age:.0f} ans | Glycémie : {args.glucose} mg/dL | IMC : {args.bmi or 'N/A'}")
    print(f" Cliché radiologique analysé : {image_path.name}")
    print("-" * 60)

    p_img = agent_image(image_path)
    p_clin = agent_clinique(clinical)
    p_final, niveau, c_img, c_clin, intercept = agent_decision(p_img, p_clin)

    print(f" 1. Agent Vision Médicale (CNN MobileNetV2) :")
    print(f"    -> Probabilité Pneumonie = {p_img:6.1%} [{'SUSPICION OPACITÉS' if p_img >= 0.5 else 'CHAMPS DÉGAGÉS'}]")
    print(f" 2. Agent Clinique (Random Forest) :")
    print(f"    -> Probabilité Risque AVC = {p_clin:6.1%} [{'RISQUE ÉLEVÉ' if p_clin >= 0.5 else 'RISQUE MODÉRÉ/FAIBLE'}]")
    print(f" 3. Agent Décisionnel (Régression Logistique) :")
    print(f"    -> Formule : z = ({c_img:.2f} × P_img) + ({c_clin:.2f} × P_clin) + ({intercept:.2f})")
    print(f"    -> Score d'Urgence Global = {p_final:6.1%}")
    print("=" * 60)
    print(f" DÉCISION FINALE DE TRIAGE : {niveau}")
    print("=" * 60)
    print(" DISCLAIMER : Ce système est un prototype de recherche universitaire.")
    print(" Il ne remplace en aucun cas un diagnostic médical certifié.\n")


if __name__ == "__main__":
    main()
