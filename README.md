# Système Multi-Agent de Triage Médical

Projet 1 du cours de Data Mining / Deep Learning. Trois agents collaborent pour produire un
diagnostic préliminaire automatisé à partir d'une radiographie thoracique et de données cliniques.

> Projet pédagogique : ne constitue pas un outil de diagnostic médical.

## Architecture

```
 Radiographie ──► [Agent Image : CNN MobileNetV2] ──► P(pneumonie) ─┐
                                                                    ├─► [Agent Décision :
 Données cliniques ─► [Agent Clinique : Random Forest] ─► P(AVC) ───┘    Régression Logistique]
                                                                          │
                                                                          ▼
                                                        URGENT / À SURVEILLER / NON URGENT
```

## Structure du dépôt

```
├── agent_image/        CNN, Transfer Learning        (train_cnn.py)
├── agent_clinique/     Random Forest                 (train_rf.py)
├── agent_decision/     Régression Logistique         (train_decision.py)
├── rapport/            plan du rapport technique
├── data/               datasets (non versionnés, voir data/README.md)
├── demo.py             démonstration du système complet
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Puis télécharger les datasets en suivant `data/README.md`.

## Exécution (dans cet ordre)

```bash
python agent_image/train_cnn.py            # 1. Agent Image
python agent_clinique/train_rf.py          # 2. Agent Clinique
python agent_decision/train_decision.py    # 3. Agent Décision (utilise les sorties de 1 et 2)
python demo.py --image <radio.jpeg> --age 67 --glucose 228.7 ...   # 4. Démo
```

Les fichiers `preds_test.csv` (petits) sont versionnés : le membre qui s'occupe de l'agent
décisionnel peut donc travailler sans réentraîner les deux autres agents.

## Équipe et répartition

| Membre | Rôle | Dossier |
|---|---|---|
| A | Agent Image (CNN) | `agent_image/` |
| B | Agent Clinique (Random Forest) | `agent_clinique/` |
| C | Agent Décision + intégration + démo | `agent_decision/`, `demo.py` |

Le rapport est rédigé à trois (voir `rapport/plan_rapport.md`).

## Workflow Git

- Une branche par membre : `agent-image`, `agent-clinique`, `agent-decision`.
- Ne jamais travailler directement sur `main`.
- `git pull` avant de commencer, petits commits clairs, Pull Request pour fusionner.
