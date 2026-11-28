# Système Multi-Agent de Triage Médical — Agent 2 : Clinique

> **Branche de travail dédiée :** Agent 2 (Random Forest — Risque Clinique AVC)  
> **Auteur :** Niykven (`aniaa199a@gmail.com`)  
> **Dataset :** *Stroke Prediction Dataset* (Kaggle)

---

## 1. Description du Module
L'**Agent 2** est responsable de l'évaluation du risque neuro-vasculaire et cardiovasculaire (AVC) à partir des données cliniques et physiologiques du patient.

Ce module s'intègre dans le Système Multi-Agent global de Triage Médical (fusionné sur la branche `main`).

---

## 2. Spécifications Techniques (Voie 1 Optimisée)

* **Algorithme :** `RandomForestClassifier` (Scikit-Learn, 700 arbres régularisés : `max_depth=5`, `min_samples_leaf=10`, `min_samples_split=20`, `max_features=0.5`, `criterion='log_loss'`).
* **Gestion du déséquilibre de classe :** `class_weight='balanced_subsample'` et seuil de décision gelé par validation croisée **5-Fold OOF** (`threshold = 0.65`).
* **Prétraitement (`ColumnTransformer`) :**
  * `MissingIndicator` sur `bmi` (capture le signal clinique lié à l'IMC non renseigné) + imputation médiane (`SimpleImputer`).
  * Interactions cardio-vasculaires ciblées (`age × hypertension`, `age × heart_disease`) via `PolynomialFeatures(degree=2, interaction_only=True)`.
  * Encodage `OneHotEncoder(handle_unknown='ignore')` pour les variables qualitatives (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`).

---

## 3. Métriques de Performance Certifiées (Test Held-Out, N=1 022)

| Métrique | Valeur | Interprétation clinique |
|---|---|---|
| **ROC-AUC** | **0.851** | Discrimination globale améliorée (OOF 5-fold = 0.845). |
| **PR-AUC** | **0.245** | Gain de précision sur classe minoritaire rare (4.9 % AVC). |
| **Rappel AVC (seuil OOF 0.65)** | **62.0 %** | 31/50 AVC détectés avec calibration nettement améliorée. |
| **Spécificité (Sains)** | **89.9 %** | Réduction des fausses alertes (874/972 sains bien filtrés). |
| **F1-Score AVC (Classe cible)** | **34.6 %** | Gain significatif (+9 pts vs baseline) au seuil OOF gelé `0.65`. |
| **Brier Score** | **0.142** | Probabilités mieux calibrées pour la fusion Agent 3. |

---

## 4. Structure de la Branche

```text
├── agent_clinique/
│   ├── model_rf.joblib      # Modèle Random Forest sérialisé
│   ├── train_rf.py          # Script d'entraînement et d'évaluation
│   └── metrics.json         # Métriques certifiées exportées
├── agent_image/             # [Réservé Agent 1 : géré sur sa branche dédiée]
├── agent_decision/          # [Réservé Agent 3 : géré sur sa branche dédiée]
├── tests/
│   └── test_agent_clinique.py # Banc de tests automatisés de l'Agent 2
├── data/
│   └── README.md            # Spécifications des datasets Kaggle
├── requirements.txt         # Dépendances Python
└── README.md                # Documentation de l'Agent 2
```

---

## 5. Exécution des Tests Unitaires

```bash
py -m unittest tests/test_agent_clinique.py -v
```
*(Résultat : 6 tests réussis sur 6 — conformité 100 %).*
