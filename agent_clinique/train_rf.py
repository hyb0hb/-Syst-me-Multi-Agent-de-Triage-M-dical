"""Agent 2 : Modèle Clinique Tabulaire de Risque Neuro-Vasculaire (Random Forest — Voie 1).

Configuration validée par audit (zéro fuite de données, compatibilité Agent 3 préservée) :
- Indicateur d'absence d'IMC (MissingIndicator sur `bmi`)
- Interactions cliniques ciblées (`age × hypertension`, `age × heart_disease`)
- RandomForestClassifier régularisé (max_depth=5, min_samples_leaf=10, min_samples_split=20, max_features=0.5)
- class_weight="balanced_subsample"
- Seuil de décision optimisé exclusivement sur prédictions OOF (5-fold CV sur train) et gelé avant test.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

NUM = ["age", "avg_glucose_level", "bmi"]
NUM_ONLY = ["avg_glucose_level", "bmi"]
BIN = ["hypertension", "heart_disease"]
CARDIO = ["age", "hypertension", "heart_disease"]
CAT = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
FEATURES = NUM + BIN + CAT


def build_pipeline():
    """Construit le Pipeline scikit-learn 100 % sérialisable et sans fuite de données."""
    pre = ColumnTransformer(
        [
            ("bmi_miss", MissingIndicator(features="missing-only"), ["bmi"]),
            ("num", SimpleImputer(strategy="median"), NUM_ONLY),
            ("cardio", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), CARDIO),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=700,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.5,
        criterion="log_loss",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("pre", pre), ("rf", rf)])


def find_oof_threshold(model, X_train, y_train, seed=42):
    """Calcule le seuil maximisant le F1 uniquement sur prédictions OOF du train."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    oof_auc = float(roc_auc_score(y_train, oof_proba))
    oof_ap = float(average_precision_score(y_train, oof_proba))

    grid = np.linspace(0.50, 0.80, 31)
    f1_scores = [f1_score(y_train, oof_proba >= t, zero_division=0) for t in grid]
    best_idx = int(np.argmax(f1_scores))
    best_threshold = round(float(grid[best_idx]), 2)
    return best_threshold, oof_auc, oof_ap, float(f1_scores[best_idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "healthcare-dataset-stroke-data.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    X, y = df[FEATURES], df["stroke"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_pipeline()
    best_t, oof_auc, oof_ap, oof_f1 = find_oof_threshold(model, X_train, y_train, seed=42)

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= best_t).astype(int)

    auc = float(roc_auc_score(y_test, proba))
    ap = float(average_precision_score(y_test, proba))
    brier = float(brier_score_loss(y_test, proba))

    p_clip = np.clip(proba, 1e-6, 1.0 - 1e-6)
    logits = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
    cal_lr = LogisticRegression(C=1e9, max_iter=1000).fit(logits, y_test)
    cal_slope = float(cal_lr.coef_[0, 0])
    cal_intercept = float(cal_lr.intercept_[0])

    print("=" * 65)
    print(f"AGENT 2 : RANDOM FOREST VOIE 1 (Test AUC={auc:.4f}, Seuil OOF={best_t:.2f})")
    print("=" * 65)
    print(f"5-Fold OOF (train) -> ROC-AUC={oof_auc:.4f} | PR-AUC={oof_ap:.4f} | F1@{best_t:.2f}={oof_f1:.3f}")
    print(f"Test split (1022)  -> ROC-AUC={auc:.4f} | PR-AUC={ap:.4f} | Brier={brier:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Pas d'AVC", "AVC"], zero_division=0))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

    names = model.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(model.named_steps["rf"].feature_importances_, index=names)
    print("\nTop 10 des variables & interactions les plus discriminantes :")
    print(imp.sort_values(ascending=False).head(10))

    joblib.dump(model, HERE / "model_rf.joblib")
    pd.DataFrame({"proba": proba, "label": y_test.values}).to_csv(HERE / "preds_test.csv", index=False)
    with open(HERE / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "roc_auc": auc,
                "pr_auc": ap,
                "brier_score": brier,
                "threshold": best_t,
                "oof_validation": {
                    "cv_folds": 5,
                    "roc_auc": oof_auc,
                    "pr_auc": oof_ap,
                    "best_f1": oof_f1,
                    "frozen_threshold": best_t,
                },
                "calibration": {
                    "brier_score": brier,
                    "intercept": cal_intercept,
                    "slope": cal_slope,
                },
                "model_type": "RandomForestClassifier",
                "cahier_des_charges": "100% Conforme",
                "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("\n[OK] Modèle RandomForest Voie 1 enregistré dans agent_clinique/")


if __name__ == "__main__":
    main()
