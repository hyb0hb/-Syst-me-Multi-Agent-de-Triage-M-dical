# Système Multi-Agent de Triage Médical

Projet d'Intelligence Artificielle et Systèmes Multi-Agents (Data Mining & Deep Learning).
Trois agents spécialisés collaborent pour produire une orientation de triage hospitalier automatisée combinant imagerie médicale et données cliniques.

> **Avertissement déontologique :** Prototype universitaire à visée de recherche et de démonstration. Ne constitue pas un dispositif médical validé.

---

## Architecture Générale

```
  Radiographie ──────► [Agent 1 : Vision CNN MobileNetV2] ─────► P(pneumonie) ─┐
                                                                               ├─► [Agent 3 : Consensus & Triage
  Données cliniques ─► [Agent 2 : Clinique Random Forest] ────► P(AVC) ────────┘    Régression Logistique L2]
                                                                                     │
                                                                                     ▼
                                                                     URGENT / À SURVEILLER / NON URGENT
                                                                    (Priorités SAU inspirées de MTS / ESI)
```

---

## Structure du Dépôt

```
├── agent_image/        Agent 1 : CNN MobileNetV2 Transfer Learning (train_cnn.py, model_cnn.keras)
├── agent_clinique/     Agent 2 : Random Forest avec ColumnTransformer (train_rf.py, model_rf.joblib)
├── agent_decision/     Agent 3 : Régression Logistique Superviseur (train_decision.py, model_lr.joblib)
├── tests/              Banc de tests automatisés (test_system.py)
├── rapport/            Rapport technique complet (rapport_technique.md)
├── static/             Frontend Web réactif (index.html, style.css, app.js, samples/)
├── data/               Instructions pour télécharger les datasets Kaggle bruts (README.md)
├── demo.py             Démonstrateur en ligne de commande (CLI)
├── server.py           Serveur API REST FastAPI + Interface Web
└── requirements.txt    Dépendances Python du projet
```

---

## Installation Rapide

```bash
# 1. Cloner et entrer dans le dossier
git clone <url-du-depot>
cd -Syst-me-Multi-Agent-de-Triage-M-dical

# 2. Créer l'environnement virtuel (Python 3.10 à 3.13)
py -m venv .venv
.venv\Scripts\activate          # Sur Linux/macOS : source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

> **Note :** Les modèles entraînés (`.keras`, `.joblib`), les métriques certifiées (`metrics.json`) et les images d'exemples (`static/samples/`) sont **déjà inclus dans le dépôt**. Vous pouvez exécuter les tests et lancer la démo immédiatement sans avoir à télécharger les volumineux datasets bruts.

---

## Utilisation et Démonstration

### 1. Lancer la Suite de Tests Automatisés
```bash
py tests/test_system.py
```
*Valide automatiquement 15 points critiques : équilibre des poids, cohérence des quadrants cliniques et sécurité SAU, filtrage physiologique, robustesse des endpoints REST FastAPI.*

### 2. Exécuter la Démonstration en Ligne de Commande (CLI)
```bash
# Exemple 1 : Suspicion d'urgence respiratoire
py demo.py --sample pneumonia --age 54 --gender Male --glucose 98.4 --bmi 26.2

# Exemple 2 : Suspicion d'AVC avec constantes tabulaires
py demo.py --sample normal --age 72 --gender Female --hypertension 1 --heart-disease 1 --glucose 215.3 --bmi 34.8
```

### 3. Lancer l'Application Web Interactive
```bash
py server.py
```
Puis ouvrez votre navigateur sur : **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.
- Interface réactive et rassurante inspirée de Qare.fr.
- 4 dossiers patients préenregistrés pour tester immédiatement les scénarios cliniques.
- Formulaire interactif avec curseurs dynamiques et téléversement de radiographies.
- Visualisation en temps réel de la jauge d'urgence et des délibérations de chaque agent.

### 4. Lancer avec Docker (Une seule commande avec Hot-Reload)

Pour exécuter toute l'application dans un conteneur isolé tout en continuant à modifier votre code en direct depuis votre éditeur (VS Code) :

```bash
# Recommandé : lancement avec Docker Compose
docker compose up

# Ou avec docker run classique (montage du code local en direct)
docker build -t medtriage-ai .
docker run -p 8000:8000 -v ${PWD}:/app medtriage-ai
```
L'interface est accessible sur **[http://localhost:8000](http://localhost:8000)**. Toute modification apportée à vos fichiers Python, HTML ou CSS est répercutée en direct sans avoir à reconstruire le conteneur.

---

## Réentraînement des Modèles (Optionnel)

Si vous téléchargez les datasets complets conformément à `data/README.md` :

```bash
# 1. Réentraînement de l'Agent Image (Recommandé sur GPU ou Google Colab)
py agent_image/train_cnn.py

# 2. Réentraînement de l'Agent Clinique (Rapide sur CPU)
py agent_clinique/train_rf.py

# 3. Calibrage de l'Agent Décisionnel par quadrants équilibrés
py agent_decision/train_decision.py
```

---

## Livrable Académique

Le rapport technique complet et structuré est consultable dans :
👉 **[rapport/rapport_technique.md](rapport/rapport_technique.md)**
