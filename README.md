# Système Multi-Agent de Triage Médical — Agent 2 : Clinique

> **Branche de travail dédiée :** Agent 2 (Random Forest — Risque Clinique AVC)  
> **Auteur :** Niykven (`aniaa199a@gmail.com`)  
> **Dataset :** *Stroke Prediction Dataset* (Kaggle)

---

## 1. Description du Module
L'**Agent 2** est responsable de l'évaluation du risque neuro-vasculaire et cardiovasculaire (AVC) à partir des données cliniques et physiologiques du patient.

Ce module s'intègre dans le Système Multi-Agent global de Triage Médical (fusionné sur la branche `main`).

---

## 2. Spécifications Techniques

* **Algorithme :** `RandomForestClassifier` (Scikit-Learn, 100 arbres de décision, profondeur max contrôlée).
* **Gestion du déséquilibre de classe :** Configuration `class_weight='balanced'` pour maximiser la sensibilité sur les cas d'AVC rares (4.9 % de prévalence).
* **Prétraitement :**
  * Imputation de la médiane pour l'IMC (`bmi`).
  * Encodage One-Hot pour les variables qualitatives (`gender`, `work_type`, `Residence_type`, `smoking_status`).
  * Mise à l'échelle standard (`StandardScaler`) des variables continues.

---

## 3. Métriques de Performance Certifiées

| Métrique | Valeur | Interprétation clinique |
|---|---|---|
| **ROC-AUC** | **0.846** | Excellente capacité de discrimination globale. |
| **Rappel (Sensibilité AVC)** | **80.0 %** | Détection prioritaire de 4 AVC sur 5 (sécurité du patient). |
| **Spécificité (Sains)** | **76.9 %** | Filtrage efficace des profils ambulatoires non urgents. |
| **F1-Score Global Pondéré** | **83.4 %** | Haute performance moyenne sur l'ensemble de la population. |
| **F1-Score AVC (Classe cible)** | **25.4 %** | Compromis mathématique lié à la rareté de la maladie (4.9 % de prévalence). |

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
*(Résultat : 5 tests réussis sur 5 — conformité 100 %).*
