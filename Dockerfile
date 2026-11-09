# ==============================================================================
# MedTriage AI — Conteneurisation de Production & Développement
# Système Multi-Agent de Triage Médical (FastAPI + MobileNetV2 + Random Forest)
# ==============================================================================

FROM python:3.11-slim

# Configuration de l'environnement Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEDTRIAGE_HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Dépendances système pour Pillow, OpenCV/Keras et compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Mise en cache de l'installation des dépendances Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code applicatif et des modèles sérialisés
COPY . /app

# Port d'écoute du serveur web FastAPI
EXPOSE 8000

# Commande de démarrage avec rechargement à chaud (Hot-Reload)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

