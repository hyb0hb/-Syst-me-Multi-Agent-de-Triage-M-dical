# Système Multi-Agent de Triage Médical (MedTriage AI)

Système d'Intelligence Artificielle et d'Aide à la Décision Médicale (Data Mining, Deep Learning & Systèmes Multi-Agents).  
Trois agents spécialisés collaborent pour produire une orientation de triage hospitalier automatisée, explicable et sécurisée, combinant imagerie thoracique et données cliniques.

> **Avertissement déontologique :** Prototype universitaire à visée de recherche et de démonstration académique. Ne constitue pas un dispositif médical certifié pour une utilisation in vivo.

---

## 🏥 Architecture Générale

```
  Cliché radiologique ──► [Agent 1 : Vision CNN MobileNetV2] ─────► P(pneumonie) ──┐
                                                                                   ├─► [Agent 3 : Consensus SAU]
  Constantes patient  ──► [Agent 2 : Clinique Random Forest] ────► P(AVC) ─────────┘    Régression Logistique OOF
                                                                                         │ + Principe de précaution
                                                                                         ▼
                                                                         URGENT / À SURVEILLER / NON URGENT
                                                                       (Priorités inspirées des échelles MTS / ESI)
```

---

## 🤖 Description des 3 Agents

### Agent 1 — Vision par Ordinateur (CNN)
- **Architecture** : **MobileNetV2** avec Transfer Learning (ImageNet) fine-tuné sur clichés thoraciques.
- **Tâche** : Détection de foyers infectieux pulmonaires et d'opacités alvéolaires (pneumonie).
- **Dataset** : *Chest X-Ray Images (Pneumonia)* — Kaggle (5 216 clichés d'entraînement, 624 clichés de test).
- **Performances test** : **ROC-AUC = 0.965** | **Rappel Pneumonie = 98.5%** (384 cas détectés sur 390, seulement 6 faux négatifs).

### Agent 2 — Risque Neuro-Vasculaire (Random Forest — Voie 1)
- **Architecture** : **Random Forest** régularisé avec pipeline sans fuite (`MissingIndicator` sur l'IMC + `SimpleImputer` (médiane) + interactions `âge × hypertension` et `âge × cardiopathie` + `OneHotEncoder`).
- **Tâche** : Détection précoce du risque d'Accident Vasculaire Cérébral (AVC) à partir des constantes tabulaires.
- **Dataset** : *Stroke Prediction Dataset* — Kaggle (5 110 patients, découpage 80/20 canonique stratifié).
- **Seuil décisionnel** : **0.65**, choisi par validation croisée 5-fold OOF (maximise le F1 sur le train) et gelé avant le test. Les probabilités ne sont pas calibrées.
- **Performances test (1 022 patients)** : **ROC-AUC = 0.851** | **Rappel AVC = 62.0%** (31/50) | **Précision = 24.0%** | **Spécificité = 89.9%** (98 faux positifs). Au seuil 0.70, le rappel était de 36 à 40% (mesure conservée dans l'historique Git, commit 5a92a42) : le gain vient du changement de seuil, pas du modèle.

### Agent 3 — Consensus Décisionnel & Triage (Régression Logistique OOF)
- **Architecture** : **Régression Logistique L2** (`class_weight='balanced'`) entraînée par late-stacking 5-fold OOF sur des paires synthétiques équilibrées de 4 quadrants (2 000 par quadrant, valeur par défaut de `train_decision.py`).
- **Poids réels appris** :
  $$z = (2.75 \times P_{\text{img}}) + (3.51 \times P_{\text{clin}}) - 2.91$$
- **Principe de Précaution SAU (Emergency Safety Net)** :
  $$P_{\text{triage}} = \max\left(P_{\text{composite}}, P_{\text{img}}, \text{align}(P_{\text{clin}})\right)$$
  Dès que l'Agent 2 détecte un risque d'AVC ($P_{\text{clin}} \ge 0.65$), le score est réaligné à $\ge 0.70$ (**URGENT**). Un AVC dont le score clinique reste sous 0.65 n'est pas classé URGENT : c'est le cas de 19 des 50 AVC du jeu de test (8 en NON URGENT, 11 en À SURVEILLER).
- **Performances (benchmark synthétique)** : **ROC-AUC = 0.877** | **PR-AUC = 0.955** | **Rappel Cas Urgents = 91.5%** | **Exactitude = 84.6%**, mesurés sur 4 000 paires synthétiques (1 000 par quadrant, 75% de cas urgents par construction), non transposables à une population réelle.

---

## 📊 Tableau Synthétique des Performances Mesurées

| Agent | Modèle | Jeu d'évaluation | ROC-AUC | Rappel Critique | Précision | Exactitude |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Agent 1** (Imagerie) | MobileNetV2 (CNN) | Test Kaggle (N=624) | **0.965** | **98.5%** *(Pneumonie)* | 82.4% | 85.9% |
| **Agent 2** (Clinique) | Random Forest Voie 1 | 20% du dataset (N=1 022) | **0.851** | **62.0%** *(AVC)* | 24.0% | 88.6% |
| **Agent 3** (Consensus) | Régression Logistique + SAU | 4 000 paires synthétiques (75% de positifs) | **0.877** | **91.5%** *(Urgences)* | 88.4% | 84.6% |

Les trois lignes ne sont pas comparables : jeux d'évaluation et prévalences différents.

> **Principe médical fondamental :** Le **rappel** est la priorité absolue aux urgences. Un faux négatif (pathologie grave non diagnostiquée) engage le pronostic vital, alors qu'un faux positif n'est pas anodin (surcharge et délais aux urgences) : voir Limites connues.

---

## ⚠️ Limites connues
- **Cohortes non appariées** : les radios (enfants, Kaggle) et les dossiers AVC (adultes) concernent des patients différents. Le consensus de l'Agent 3 est évalué sur des paires synthétiques (75% de cas urgents par construction) : ces chiffres ne sont pas transposables à une population réelle.
- **Agent 1** : rappel NORMAL de 65% au seuil 0.5 (82 radios normales sur 234 classées pneumonie), précision pneumonie 82.4%. Sa validation est découpée par image et non par patient : une simulation donne environ 40% d'images de validation partageant un identifiant patient (déduit des noms de fichiers) avec l'entraînement, donc des métriques de validation optimistes ; le dossier test/ n'a aucun identifiant commun avec train/.
- **Agent 2** : rappel 62.0% (31/50) mais précision 24.0% (prévalence de 4.9%) ; 19 AVC sur 50 ne sont pas classés URGENT. Les probabilités affichées ne sont pas calibrées : « 54% » ne correspond pas à 54% de risque réel.
- **Faux positifs** : un faux positif aux urgences a un coût (surcharge, délais). Le seuil URGENT du système favorise le rappel au détriment de la précision.
- **Entrées** : sans image fournie, le système utilise une radio normale par défaut ; une image hors sujet reçoit tout de même un score.

---

## 📁 Structure du Projet

```
.
├── agent_image/            # Agent 1 : Vision par Ordinateur
│   ├── model_cnn.keras     # Poids réels du modèle MobileNetV2 (20.88 Mo)
│   ├── metrics.json        # Métriques certifiées du CNN
│   └── train_cnn.py        # Script de fine-tuning reproductible
├── agent_clinique/         # Agent 2 : Risque Neuro-vasculaire
│   ├── model_rf.joblib     # Pipeline Random Forest optimisé (2.31 Mo)
│   ├── metrics.json        # Métriques certifiées de l'Agent 2
│   └── train_rf.py         # Script d'entraînement 5-fold OOF Voie 1
├── agent_decision/         # Agent 3 : Superviseur de Consensus
│   ├── model_lr.joblib     # Modèle binaire de Régression Logistique (879 o)
│   ├── metrics.json        # Métriques certifiées et formule dynamique
│   └── train_decision.py   # Script de late-stacking OOF 4-quadrants
├── backend/                # API REST FastAPI modulaire
│   ├── main.py             # Point d'entrée de l'application FastAPI
│   ├── config.py           # Configuration centralisée des chemins et constantes
│   ├── routers/            # Endpoints API (/triage, /presets, /metrics, /history)
│   ├── schemas/            # Validation des schémas d'entrée/sortie Pydantic
│   └── services/           # Logique métier (triage, modèles, historique)
├── static/                 # Interface Web interactive
│   ├── index.html          # Dashboard médical responsive (sans hardcoding)
│   ├── style.css           # Thème médical hospitalier
│   ├── app.js              # Logique UI dynamique et appels API en temps réel
│   └── samples/            # Clichés radiologiques types (normal.jpg, pneumonia.jpg)
├── tests/                  # Banc de validation automatisé
│   └── test_system.py      # 23 tests unitaires et d'intégration
├── data/                   # Datasets Kaggle et historique des requêtes
├── demo.py                 # Démonstrateur CLI autonome
├── server.py               # Script de lancement avec option --port
└── requirements.txt        # Dépendances Python épinglées
```

---

## 🚀 Installation & Démarrage

### 1. Prérequis
- Python 3.10 à 3.13 installé.
- Git.

### 2. Cloner et installer
```bash
# Cloner le dépôt
git clone https://github.com/hyb0hb/-Syst-me-Multi-Agent-de-Triage-M-dical.git
cd -Syst-me-Multi-Agent-de-Triage-M-dical

# Créer et activer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell / CMD)
# source .venv/bin/activate   # Linux / macOS

# Installer les dépendances
pip install -r requirements.txt
```

> **Note :** Les modèles entraînés (`.keras`, `.joblib`) et les radiographies d'exemples sont **déjà inclus dans le dépôt**. Aucun entraînement n'est requis pour exécuter le système.

---

## 💻 Utilisation

### 1. Lancer l'Interface Web & l'API REST
```bash
# Lancement direct sur le port 8000 par défaut
python server.py

# Ou sur un port personnalisé (ex: 8001)
python server.py --port 8001
```
Accédez au tableau de bord : **http://127.0.0.1:8000** (ou 8001 avec `--port 8001`).

- **Presets cliniques** : testez en 1 clic les 4 cas types (détresse respiratoire, AVC aigu, choc mixte, routine).
- **Mode aléatoire** : pioche un patient réel depuis le dataset de test.
- **Formulaire complet** : téléversez n'importe quelle radiographie thoracique (JPEG/PNG) et ajustez les constantes physiologiques.

### 2. Démonstrateur en Ligne de Commande (CLI)
Le script [demo.py](demo.py) s'exécute directement sans serveur web :
```bash
# Cas 1 : Détresse respiratoire aiguë (pneumonie)
python demo.py --sample pneumonia --age 54 --gender Male --glucose 98.4 --bmi 26.2

# Cas 2 : Suspicion d'AVC avec cliché pulmonaire sain
python demo.py --sample normal --age 72 --gender Female --hypertension 1 --heart-disease 1 --glucose 215.3 --bmi 34.8

# Mode aléatoire sur le dataset de test
python demo.py --random 3
```

### 3. Exécuter la Suite de Tests (23/23 Tests)
```bash
python -m unittest discover -s tests -v
```
Vérifie 23 points critiques : chargement et cohérence des 3 modèles, seuils et paliers de triage, rejet des valeurs aberrantes (âges négatifs, IMC irréalistes), parité stricte Backend/CLI et endpoints FastAPI.

---

## 🌐 Endpoints de l'API REST

| Méthode | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/triage` | Inférence complète des 3 agents (multipart/form-data avec image et constantes). |
| `GET` | `/api/presets` | Retourne les 4 cas cliniques types préenregistrés. |
| `GET` | `/api/metrics` | Retourne les métriques certifiées dynamiques des 3 agents et la formule de consensus. |
| `GET` | `/api/history` | Historique des évaluations de triage réalisées. |
| `GET` | `/api/health` | Vérification de l'état opérationnel du serveur. |

---

## 🔄 Réentraînement des Modèles (Reproductibilité)

Tous les scripts sont 100 % reproductibles :
```bash
# 1. Réentraîner l'Agent 1 (CNN MobileNetV2)
python agent_image/train_cnn.py

# 2. Réentraîner l'Agent 2 (Random Forest Voie 1 - 5-fold OOF)
python agent_clinique/train_rf.py

# 3. Réentraîner l'Agent 3 (Régression Logistique OOF - Consensus SAU)
python agent_decision/train_decision.py
```

---

## 📚 Datasets Utilisés

1. **Chest X-Ray Images (Pneumonia)** — [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) : 5 856 radiographies thoraciques pédiatriques (enfants de 1 à 5 ans) réparties en classes Normal et Pneumonie.
2. **Stroke Prediction Dataset** — [Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) : 5 110 observations cliniques (âge, hypertension, cardiopathie, glycémie, IMC, tabagisme) avec diagnostic avéré d'AVC.
