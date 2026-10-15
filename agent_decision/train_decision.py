
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def make_pairs(img, clin, n, rng):
    
    i = rng.integers(0, len(img), n)
    j = rng.integers(0, len(clin), n)
    X = np.column_stack([img["proba"].values[i], clin["proba"].values[j]])
    y = ((img["label"].values[i] == 1) | (clin["label"].values[j] == 1)).astype(int)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default=str(ROOT / "agent_image" / "preds_test.csv"))
    parser.add_argument("--clin", default=str(ROOT / "agent_clinique" / "preds_test.csv"))
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-test", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    img = pd.read_csv(args.img)
    clin = pd.read_csv(args.clin)
    rng = np.random.default_rng(args.seed)

    # Séparation stricte : moitié A pour entraîner, moitié B pour évaluer
    img_a, img_b = train_test_split(img, test_size=0.5, stratify=img["label"], random_state=args.seed)
    clin_a, clin_b = train_test_split(clin, test_size=0.5, stratify=clin["label"], random_state=args.seed)

    X_train, y_train = make_pairs(img_a, clin_a, args.n_train, rng)
    X_test, y_test = make_pairs(img_b, clin_b, args.n_test, rng)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    print(classification_report(y_test, y_pred, target_names=["Non urgent", "Urgent"], zero_division=0))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
    auc = roc_auc_score(y_test, proba)
    print(f"ROC-AUC (décision finale) : {auc:.3f}")
    print(f"ROC-AUC radio seule       : {roc_auc_score(y_test, X_test[:, 0]):.3f}")
    print(f"ROC-AUC clinique seule    : {roc_auc_score(y_test, X_test[:, 1]):.3f}")
    print(f"Coefficients : radio={model.coef_[0][0]:.2f}, clinique={model.coef_[0][1]:.2f}, "
          f"biais={model.intercept_[0]:.2f}")

    joblib.dump(model, HERE / "model_lr.joblib")
    with open(HERE / "metrics.json", "w") as f:
        json.dump(
            {"roc_auc": auc,
             "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)},
            f, indent=2,
        )
    print("Sauvegardé : model_lr.joblib, metrics.json")


if __name__ == "__main__":
    main()
