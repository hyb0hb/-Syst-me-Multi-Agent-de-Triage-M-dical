
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

NUM = ["age", "avg_glucose_level", "bmi"]
BIN = ["hypertension", "heart_disease"]
CAT = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
FEATURES = NUM + BIN + CAT


def build_pipeline():
    """Prétraitement + Random Forest dans un seul objet (réutilisable tel quel dans la démo)."""
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), NUM),
            ("bin", "passthrough", BIN),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",  
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("rf", rf)])


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
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    print(classification_report(y_test, y_pred, target_names=["Pas d'AVC", "AVC"], zero_division=0))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
    auc = roc_auc_score(y_test, proba)
    print(f"ROC-AUC test : {auc:.3f}")

    
    names = model.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(model.named_steps["rf"].feature_importances_, index=names)
    print("\nTop 10 des variables importantes :\n", imp.sort_values(ascending=False).head(10))

    joblib.dump(model, HERE / "model_rf.joblib")
    pd.DataFrame({"proba": proba, "label": y_test.values}).to_csv(HERE / "preds_test.csv", index=False)
    with open(HERE / "metrics.json", "w") as f:
        json.dump(
            {"roc_auc": auc,
             "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)},
            f, indent=2,
        )
    print("Sauvegardé : model_rf.joblib, preds_test.csv, metrics.json")


if __name__ == "__main__":
    main()
