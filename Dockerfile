# Étape 1: Utiliser une image de base Python
FROM python:3.9-slim

# Étape 2: Définir le répertoire de travail
WORKDIR /app

# Étape 3: Copier les fichiers nécessaires dans l'image Docker
COPY . /app

# Étape 4: Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5: Exposer le port
EXPOSE 9000

# Étape 6: Commande pour lancer l'application avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
