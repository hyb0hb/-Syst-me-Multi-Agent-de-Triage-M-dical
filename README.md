# Système Multi-Agent de Triage Médical

Projet d'Intelligence Artificielle et Systèmes Multi-Agents (Data Mining & Deep Learning).  
Trois agents spécialisés collaborent pour produire une orientation de triage hospitalier automatisée combinant imagerie médicale et données cliniques.

> **Avertissement déontologique :** Prototype universitaire à visée de recherche et de démonstration. Ne constitue pas un dispositif médical validé.

---

## Architecture Générale

```
  Radiographie ──────► [Agent 1 : Vision CNN MobileNetV2] ─────► P(pneumonie) ─┐
                                                                                ├─► [Agent 3 : Consensus & Triage]
  Données cliniques ─► [Agent 2 : Clinique Random Forest] ────► P(AVC) ────────┘    Régression Logistique OOF
                                                                                      │
                                                                                      ▼
                                                                      URGENT / À SURVEILLER / NON URGENT
                                                                     (Priorités SAU inspirées de MTS / ESI)
```

---

## Agents

### Agent 1 — Vision par Ordinateur (CNN)
- Architecture : **MobileNetV2** avec Transfer Learning (ImageNet)
- Tâche : Détection de pneumonie sur radiographie thoracique
- Dataset : Chest X-Ray Images (Kaggle) — 5 216 images
- Recall : **0.985** sur le jeu de test

### Agent 2 — Analyse Clinique (Random Forest)
- Architecture : **Random Forest** avec ColumnTransformer (StandardScaler + OneHotEncoder)
- Tâche : Estimation du risque d'AVC à partir des constantes cliniques
- Dataset : Stroke Prediction Dataset (Kaggle) — 5 110 patients
- Variables : âge, IMC, glycémie, genre, hypertension, cardiopathie, tabagisme
- Recall : **0.620** sur le jeu de test

### Agent 3 — Décision & Fusion (Régression Logistique)
- Architecture : **Régression Logistique L2** entraînée par validation croisée OOF
- Tâche : Fusion des probabilités des Agents 1 et 2 → catégorie de triage finale
- Méthode OOF : prédictions hors-échantillon pour éviter toute fuite de données
- Seuil clinique aligné à 0.70 pour maximiser le recall médical
- Recall système : **0.620** | ROC-AUC : **0.851** (test held-out, N=1 022, 50 AVC)

---

## Structure du Dépôt

```
├── agent_image/        Agent 1 — CNN MobileNetV2 (train_cnn.py, model_cnn.keras)
├── agent_clinique/     Agent 2 — Random Forest (train_rf.py, model_rf.joblib)
├── agent_decision/     Agent 3 — LR OOF (train_experiment_agent3.py, model_lr_oof.joblib)
├── backend/            API REST FastAPI (services, routers, schemas)
├── models/             Modèles entraînés et artefacts expérimentaux
├── static/             Frontend Web (index.html, style.css, app.js)
├── tests/              Suite de 33 tests automatisés (test_system.py)
├── data/               Datasets (non inclus — voir ci-dessous)
├── demo.py             Démonstrateur CLI
├── server.py           Serveur FastAPI + Interface Web
└── requirements.txt    Dépendances Python
```

---

## Installation Rapide

```bash
# 1. Cloner le dépôt
git clone https://github.com/hyb0hb/-Syst-me-Multi-Agent-de-Triage-M-dical.git
cd -Syst-me-Multi-Agent-de-Triage-M-dical

# 2. Créer l'environnement virtuel (Python 3.10 à 3.13)
py -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Installer les dépendances
pip install -r requirements.txt
```

> **Note :** Les modèles entraînés (`.keras`, `.joblib`) et les images d'exemples (`static/samples/`) sont **déjà inclus**. Vous pouvez lancer les tests et la démo immédiatement sans télécharger les datasets.

---

## Utilisation

### 1. Tests automatisés
```bash
py tests/test_system.py
```
Valide 33 points critiques : séparation train/test, intégrité des modèles, cohérence des endpoints REST, absence de fuite de données.

### 2. Démo en ligne de commande (CLI)
```bash
# Suspicion d'urgence respiratoire
py demo.py --sample pneumonia --age 54 --gender Male --glucose 98.4 --bmi 26.2

# Suspicion d'AVC
py demo.py --sample normal --age 72 --gender Female --hypertension 1 --heart-disease 1 --glucose 215.3 --bmi 34.8
```

### 3. Interface Web
```bash
py server.py
```
Ouvrez **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

- 4 dossiers patients préenregistrés pour tester les scénarios cliniques
- Formulaire interactif avec téléversement de radiographies
- Visualisation en temps réel de la jauge d'urgence et des délibérations de chaque agent

### 4. Docker
```bash
docker compose up
```
Interface accessible sur **[http://localhost:8000](http://localhost:8000)**

---

## Réentraînement des Modèles (Optionnel)

Téléchargez les datasets Kaggle dans `data/` avant de lancer :

```bash
# Agent 1 — CNN (recommandé sur GPU ou Google Colab)
py agent_image/train_cnn.py

# Agent 2 — Random Forest
py agent_clinique/train_rf.py

# Agent 3 — Régression Logistique OOF
py agent_decision/train_experiment_agent3.py
```

---

## Performances

| Agent | Métrique principale | Valeur |
|---|---|---|
| Agent 1 — CNN | Recall (pneumonie) | **0.985** |
| Agent 2 — RF | Recall (AVC) | **0.620** |
| Agent 3 — Système complet | Recall (AVC, test held-out) | **0.620** |
| Agent 3 — Système complet | ROC-AUC | **0.851** |

> Le recall est la métrique prioritaire en contexte médical d'urgence.  
> Un faux négatif (AVC non détecté) est plus grave qu'un faux positif (examen supplémentaire).

---

## Datasets utilisés

| Dataset | Source | Taille |
|---|---|---|
| Chest X-Ray Images (Pneumonia) | [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) | 5 216 images |
| Stroke Prediction Dataset | [Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) | 5 110 patients |

---

*Prototype académique — pas un dispositif médical. Résultats non validés cliniquement.*
