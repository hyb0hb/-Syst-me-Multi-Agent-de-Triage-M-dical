"""Point d'entrée du serveur MedTriage AI.

Ce fichier délègue au backend modulaire FastAPI (backend/main.py).

Lancement :
    py server.py                  # Port par défaut (8000)
    py server.py --port 8001      # Port personnalisé 8001
"""

import argparse
import os
import uvicorn

# Réexporte l'application et symboles clés pour compatibilité
from backend.main import app  # noqa: F401
from backend.config import PRESETS  # noqa: F401
from backend.services.validation_service import validate_clinical_inputs  # noqa: F401

if __name__ == "__main__":
    default_host = os.environ.get("MEDTRIAGE_HOST", "127.0.0.1")
    default_port = int(os.environ.get("PORT", os.environ.get("MEDTRIAGE_PORT", 8000)))

    parser = argparse.ArgumentParser(description="MedTriage AI — Serveur Web et API REST")
    parser.add_argument("--host", type=str, default=default_host, help=f"Hôte d'écoute (défaut: {default_host})")
    parser.add_argument("-p", "--port", type=int, default=default_port, help=f"Port d'écoute (défaut: {default_port})")
    parser.add_argument("--reload", action="store_true", default=True, help="Rechargement à chaud (défaut: activé)")
    args = parser.parse_args()

    print(f"[INFO] Démarrage du serveur MedTriage AI sur http://{args.host}:{args.port}")
    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=args.reload)
