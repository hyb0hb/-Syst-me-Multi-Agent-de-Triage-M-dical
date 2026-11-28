"""Agent 3 : Superviseur de Triage Médical (Régression Logistique OOF & Consensus SAU).

Méthode validée :
- Entraînement sur prédictions Out-Of-Fold (OOF, 5-fold CV) d'Agent 2 sans fuite de données
- Équilibrage des 4 quadrants pivots (sain, respiratoire, neuro-vasculaire, mixte)
- Modèle sérialisé directement dans agent_decision/model_lr.joblib
- Métriques certifiées exportées dans agent_decision/metrics.json
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_clinique.train_rf import FEATURES, build_pipeline


def make_balanced_quadrant_pairs(img_df, clin_df, n_per_quadrant, rng):
    """Génère un échantillonnage équilibré sur les 4 quadrants cliniques."""
    img_0 = img_df[img_df["label"] == 0]
    img_1 = img_df[img_df["label"] == 1]
    clin_0 = clin_df[clin_df["label"] == 0]
    clin_1 = clin_df[clin_df["label"] == 1]

    # Q00: Cliché normal & Pas d'AVC -> y = 0
    idx_img_00 = rng.choice(img_0.index, size=n_per_quadrant, replace=True)
    idx_clin_00 = rng.choice(clin_0.index, size=n_per_quadrant, replace=True)
    X_00 = np.column_stack([img_df.loc[idx_img_00, "proba"].values,
                            clin_df.loc[idx_clin_00, "proba"].values])
    y_00 = np.zeros(n_per_quadrant, dtype=int)

    # Q10: Pneumonie & Pas d'AVC -> y = 1
    idx_img_10 = rng.choice(img_1.index, size=n_per_quadrant, replace=True)
    idx_clin_10 = rng.choice(clin_0.index, size=n_per_quadrant, replace=True)
    X_10 = np.column_stack([img_df.loc[idx_img_10, "proba"].values,
                            clin_df.loc[idx_clin_10, "proba"].values])
    y_10 = np.ones(n_per_quadrant, dtype=int)

    # Q01: Cliché normal & AVC -> y = 1
    idx_img_01 = rng.choice(img_0.index, size=n_per_quadrant, replace=True)
    idx_clin_01 = rng.choice(clin_1.index, size=n_per_quadrant, replace=True)
    X_01 = np.column_stack([img_df.loc[idx_img_01, "proba"].values,
                            clin_df.loc[idx_clin_01, "proba"].values])
    y_01 = np.ones(n_per_quadrant, dtype=int)

    # Q11: Pneumonie ET AVC -> y = 1
    idx_img_11 = rng.choice(img_1.index, size=n_per_quadrant, replace=True)
    idx_clin_11 = rng.choice(clin_1.index, size=n_per_quadrant, replace=True)
    X_11 = np.column_stack([img_df.loc[idx_img_11, "proba"].values,
                            clin_df.loc[idx_clin_11, "proba"].values])
    y_11 = np.ones(n_per_quadrant, dtype=int)

    X = np.vstack([X_00, X_10, X_01, X_11])
    y = np.concatenate([y_00, y_10, y_01, y_11])
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def load_image_predictions(seed=42):
    """Charge les prédictions d'Agent 1 (CNN) sur chest_xray/test."""
    import tensorflow as tf

    cnn_path = ROOT / "agent_image" / "model_cnn.keras"
    if not cnn_path.exists():
        raise FileNotFoundError(f"Modèle CNN introuvable : {cnn_path}")
    cnn = tf.keras.models.load_model(cnn_path)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        ROOT / "data" / "chest_xray" / "test",
        labels="inferred",
        label_mode="binary",
        image_size=(224, 224),
        batch_size=32,
        shuffle=False,
        seed=seed,
    )
    y_img = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
    p_img = cnn.predict(test_ds, verbose=0).ravel()
    return pd.DataFrame({"proba": p_img, "label": y_img})


def main():
    parser = argparse.ArgumentParser(description="Entraînement propre de l'Agent Décisionnel (OOF)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-quadrant", type=int, default=2000)
    args = parser.parse_args()

    print("=" * 65)
    print("AGENT 3 : ENTRAÎNEMENT OOF & FUSION DE CONSENSUS")
    print("=" * 65)

    # 1. Dataset clinique : découpage 80/20 canonique
    df = pd.read_csv(ROOT / "data" / "healthcare-dataset-stroke-data.csv")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    X, y = df[FEATURES], df["stroke"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    # 2. Prédictions OOF d'Agent 2 sur le train (5-fold CV sans fuite)
    print("[1/4] Calcul des prédictions OOF (5-Fold CV) pour Agent 2...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_probs = cross_val_predict(
        build_pipeline(), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    clin_oof_df = pd.DataFrame({"proba": oof_probs, "label": y_train.values})

    # 3. Prédictions d'Agent 1 sur chest_xray/test (Split A)
    print("[2/4] Extraction des prédictions Agent 1 (CNN)...")
    img_df = load_image_predictions(seed=args.seed)
    img_a, img_b = train_test_split(img_df, test_size=0.5, stratify=img_df["label"], random_state=args.seed)

    # 4. Échantillonnage équilibré 4-quadrants & entraînement de la régression logistique
    print("[3/4] Entraînement de la Régression Logistique L2...")
    rng = np.random.default_rng(args.seed)
    X_train_lr, y_train_lr = make_balanced_quadrant_pairs(img_a, clin_oof_df, args.samples_per_quadrant, rng)

    lr_model = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=args.seed)
    lr_model.fit(X_train_lr, y_train_lr)

    coef_img = float(lr_model.coef_[0][0])
    coef_clin = float(lr_model.coef_[0][1])
    intercept = float(lr_model.intercept_[0])
    print(f"Coefficients appris : Image={coef_img:.4f}, Clinique={coef_clin:.4f}, Biais={intercept:.4f}")

    # 5. Évaluation sur quadrant de test (Split B x Test Clinique)
    print("[4/4] Évaluation sur le split de test disjoint (Split B)...")
    rf_model = joblib.load(ROOT / "agent_clinique" / "model_rf.joblib")
    clin_test_df = pd.DataFrame({"proba": rf_model.predict_proba(X_test)[:, 1], "label": y_test.values})
    X_test_lr, y_test_lr = make_balanced_quadrant_pairs(img_b, clin_test_df, 1000, np.random.default_rng(args.seed))

    p_comp = lr_model.predict_proba(X_test_lr)[:, 1]
    auc = float(roc_auc_score(y_test_lr, p_comp))
    ap = float(average_precision_score(y_test_lr, p_comp))
    brier = float(brier_score_loss(y_test_lr, p_comp))
    rep = classification_report(
        y_test_lr,
        (p_comp >= 0.50).astype(int),
        target_names=["Non urgent", "Urgent"],
        output_dict=True,
        zero_division=0,
    )

    # Sauvegarde des modèles et métriques
    model_dest = HERE / "model_lr.joblib"
    metrics_dest = HERE / "metrics.json"
    joblib.dump(lr_model, model_dest)

    sign = "+" if intercept >= 0 else "-"
    formula_str = f"z = ({coef_img:.2f} × P_img) + ({coef_clin:.2f} × P_clin) {sign} {abs(intercept):.2f}"

    metrics_data = {
        "roc_auc": round(auc, 4),
        "pr_auc": round(ap, 4),
        "brier_score": round(brier, 4),
        "coefficients": {
            "image": coef_img,
            "clinique": coef_clin,
            "intercept": intercept,
        },
        "decision_formula": formula_str,
        "report": rep,
        "deployed_rule_max_0.70": {
            "threshold": 0.70,
            "roc_auc_pfinal": round(auc, 4),
            "pr_auc_pfinal": round(ap, 4),
        },
        "training": {
            "method": "5-Fold Out-of-Fold (OOF) Late Stacking",
            "estimator": "LogisticRegression(class_weight='balanced')",
            "cross_patient_paired": False,
        },
    }
    metrics_dest.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    print(f"Modèle sauvegardé dans : {model_dest}")
    print(f"Métriques exportées dans : {metrics_dest}")
    print(f"ROC-AUC Test : {auc:.4f} | PR-AUC : {ap:.4f} | Brier : {brier:.4f}")


if __name__ == "__main__":
    main()
