"""Agent 3 : Superviseur de Triage Médical (Régression Logistique de Consensus).

CADRE THÉORIQUE ET DÉONTOLOGIQUE :
==================================
1. Nature de l'Agent :
   Cet agent n'effectue PAS une "fusion multimodale conjointe intra-patient" au sens
   biologique strict, car les deux jeux de données d'origine (Chest X-Ray et Stroke Dataset)
   proviennent de cohortes indépendantes non appariées (patients pédiatriques d'un côté,
   adultes/seniors de l'autre).
   Il agit comme un **Superviseur de Triage Hiérarchique (Late Stacking Orchestrator)** :
   il intègre les probabilités a posteriori émises par deux agents spécialistes indépendants
   (Agent 1 : Vision / Pneumonie, Agent 2 : Clinique / AVC) afin de modéliser une règle
   d'orientation d'urgence composite (priorité SAU si risque respiratoire aigu OU risque neuro-vasculaire).

2. Correction du Biais de Prévalence et Équilibrage par Quadrants :
   Dans l'implémentation naïve initiale, le fort déséquilibre (62.5% de pneumonies vs 4.9% d'AVC)
   faisait que le signal radiologique écrasait le signal clinique d'un facteur 6.6, conduisant
   à sous-évaluer dramatiquement les urgences d'AVC chez les patients sans opacité pulmonaire.
   La méthode `make_balanced_quadrant_pairs` résout ce biais en calibrant la régression logistique
   sur une grille expérimentale équilibrée des 4 situations cliniques possibles :
     - Q00 : Ni pneumonie, ni AVC (y = 0, Non urgent)
     - Q10 : Pneumonie seule (y = 1, Urgent)
     - Q01 : AVC seul (y = 1, Urgent)
     - Q11 : Pneumonie ET AVC (y = 1, Urgent)

3. Prévention des Fuites de Données (Data Leakage) :
   Les paires d'entraînement et de test sont construites sur des partitions d'échantillons
   strictement disjointes (Split A pour l'apprentissage, Split B pour l'évaluation).
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def make_balanced_quadrant_pairs(img_df, clin_df, n_per_quadrant, rng):
    """Génère un échantillonnage expérimental équilibré sur les 4 quadrants cliniques.
    
    Permet à la régression logistique d'attribuer une sensibilité équitable
    aux deux filières de gravité sans que la rareté relative de l'une n'annihile son poids.
    """
    img_0 = img_df[img_df["label"] == 0]
    img_1 = img_df[img_df["label"] == 1]
    clin_0 = clin_df[clin_df["label"] == 0]
    clin_1 = clin_df[clin_df["label"] == 1]

    # Q00: Cliché normal & Pas d'AVC -> y = 0 (Ambulatoire / Non urgent)
    idx_img_00 = rng.choice(img_0.index, size=n_per_quadrant, replace=True)
    idx_clin_00 = rng.choice(clin_0.index, size=n_per_quadrant, replace=True)
    X_00 = np.column_stack([img_df.loc[idx_img_00, "proba"].values,
                            clin_df.loc[idx_clin_00, "proba"].values])
    y_00 = np.zeros(n_per_quadrant, dtype=int)

    # Q10: Pneumonie & Pas d'AVC -> y = 1 (Urgence respiratoire)
    idx_img_10 = rng.choice(img_1.index, size=n_per_quadrant, replace=True)
    idx_clin_10 = rng.choice(clin_0.index, size=n_per_quadrant, replace=True)
    X_10 = np.column_stack([img_df.loc[idx_img_10, "proba"].values,
                            clin_df.loc[idx_clin_10, "proba"].values])
    y_10 = np.ones(n_per_quadrant, dtype=int)

    # Q01: Cliché normal & AVC -> y = 1 (Urgence neuro-vasculaire)
    idx_img_01 = rng.choice(img_0.index, size=n_per_quadrant, replace=True)
    idx_clin_01 = rng.choice(clin_1.index, size=n_per_quadrant, replace=True)
    X_01 = np.column_stack([img_df.loc[idx_img_01, "proba"].values,
                            clin_df.loc[idx_clin_01, "proba"].values])
    y_01 = np.ones(n_per_quadrant, dtype=int)

    # Q11: Pneumonie ET AVC -> y = 1 (Urgence vitale mixte)
    idx_img_11 = rng.choice(img_1.index, size=n_per_quadrant, replace=True)
    idx_clin_11 = rng.choice(clin_1.index, size=n_per_quadrant, replace=True)
    X_11 = np.column_stack([img_df.loc[idx_img_11, "proba"].values,
                            clin_df.loc[idx_clin_11, "proba"].values])
    y_11 = np.ones(n_per_quadrant, dtype=int)

    X = np.vstack([X_00, X_10, X_01, X_11])
    y = np.concatenate([y_00, y_10, y_01, y_11])

    # Mélange aléatoire
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def main():
    parser = argparse.ArgumentParser(description="Entraînement de l'Agent Décisionnel de Triage")
    parser.add_argument("--img", default=str(ROOT / "agent_image" / "preds_test.csv"))
    parser.add_argument("--clin", default=str(ROOT / "agent_clinique" / "preds_test.csv"))
    parser.add_argument("--n-quadrant-train", type=int, default=1000,
                        help="Nombre d'échantillons par quadrant pour le train (total = 4 * n)")
    parser.add_argument("--n-quadrant-test", type=int, default=500,
                        help="Nombre d'échantillons par quadrant pour le test (total = 4 * n)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 65)
    print("AGENT 3 : SUPERVISEUR DE TRIAGE MÉDICAL (RÉGRESSION LOGISTIQUE)")
    print("=" * 65)

    img = pd.read_csv(args.img)
    clin = pd.read_csv(args.clin)
    rng = np.random.default_rng(args.seed)

    print(f"Chargement : Agent Image ({len(img)} preds), Agent Clinique ({len(clin)} preds)")

    # Séparation stricte et étanche en 2 moitiés disjointes (Split A apprentissage, Split B test)
    img_a, img_b = train_test_split(img, test_size=0.5, stratify=img["label"], random_state=args.seed)
    clin_a, clin_b = train_test_split(clin, test_size=0.5, stratify=clin["label"], random_state=args.seed)

    X_train, y_train = make_balanced_quadrant_pairs(img_a, clin_a, args.n_quadrant_train, rng)
    X_test, y_test = make_balanced_quadrant_pairs(img_b, clin_b, args.n_quadrant_test, rng)

    print(f"Ensemble d'entraînement : {len(X_train)} paires synthétiques équilibrées")
    print(f"Ensemble d'évaluation   : {len(X_test)} paires synthétiques étanches")

    # Modèle de consensus : Régression Logistique avec régularisation L2 modérée
    model = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=args.seed)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba))
    ap = float(average_precision_score(y_test, proba))
    brier = float(brier_score_loss(y_test, proba))
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=["Non urgent", "Urgent"], output_dict=True, zero_division=0)

    print("\n--- Performances sur le banc de test d'orchestration ---")
    print(classification_report(y_test, y_pred, target_names=["Non urgent", "Urgent"], zero_division=0))
    print(f"Matrice de confusion :\n{cm}")
    print(f"ROC-AUC           : {auc:.4f}")
    print(f"PR-AUC (Avg Prec) : {ap:.4f}")
    print(f"Score de Brier    : {brier:.4f}")

    coef_img = float(model.coef_[0][0])
    coef_clin = float(model.coef_[0][1])
    intercept = float(model.intercept_[0])

    print("\n--- Coefficients appris de la règle de consensus ---")
    print(f"  Poids Radio (P_pneumonie) : {coef_img:+.4f}")
    print(f"  Poids Clinique (P_AVC)    : {coef_clin:+.4f}")
    print(f"  Biais (Intercept)         : {intercept:+.4f}")
    print(f"  Formule Logit : z = {coef_img:.2f} * P_img + {coef_clin:.2f} * P_clin + ({intercept:.2f})")

    # Tests de cohérence clinique des cas pivots
    p_sain = float(model.predict_proba([[0.05, 0.05]])[0, 1])
    p_pneu_seule = float(model.predict_proba([[0.70, 0.05]])[0, 1])
    p_avc_seul = float(model.predict_proba([[0.05, 0.70]])[0, 1])
    p_mixte = float(model.predict_proba([[0.70, 0.70]])[0, 1])

    print("\n--- Validation des 4 scénarios cliniques pivots ---")
    print(f"  1. Patient sain (radio=0.05, avc=0.05)       -> Triage = {p_sain:.1%} (attendu : faible)")
    print(f"  2. Pneumonie seule (radio=0.70, avc=0.05)   -> Triage = {p_pneu_seule:.1%} (attendu : urgent/surveillance)")
    print(f"  3. AVC seul (radio=0.05, avc=0.70)          -> Triage = {p_avc_seul:.1%} (attendu : urgent/surveillance)")
    print(f"  4. Urgence mixte (radio=0.70, avc=0.70)      -> Triage = {p_mixte:.1%} (attendu : très urgent)")

    # Sauvegarde des artefacts
    joblib.dump(model, HERE / "model_lr.joblib")
    metrics_data = {
        "roc_auc": auc,
        "pr_auc": ap,
        "brier_score": brier,
        "coefficients": {
            "image": coef_img,
            "clinique": coef_clin,
            "intercept": intercept
        },
        "report": rep,
        "methodology": {
            "agent_type": "Hierarchical Triage Supervisor (Late Stacking Orchestration)",
            "training_sampling": "Balanced 4-Quadrant Stratified Sampling",
            "cross_patient_paired": False,
            "warning": "Prototype experimental - non valide pour diagnostic clinique in vivo."
        }
    }
    with open(HERE / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)

    print(f"\nModèle et métriques enregistrés dans {HERE.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
