"""Point d'entrée du serveur MedTriage AI.

Ce fichier est conservé pour compatibilité. Il délègue entièrement
au backend modulaire (backend/main.py).

Lancement :
    uv run uvicorn server:app --port 8000 --reload
ou :
    uv run uvicorn backend.main:app --port 8000 --reload
"""

# Réexporte l'application et symboles clés depuis le backend modulaire pour compatibilité
from backend.main import app  # noqa: F401
from backend.config import PRESETS  # noqa: F401
from backend.services.validation_service import validate_clinical_inputs  # noqa: F401

if __name__ == "__main__":
    import os
    import uvicorn
    host = os.environ.get("MEDTRIAGE_HOST", "127.0.0.1")
    uvicorn.run("backend.main:app", host=host, port=8000, reload=True)
