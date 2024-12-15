# Utiliser une image Python légère comme base
FROM python:3.9-slim

# Copier les fichiers de votre projet dans le conteneur
COPY ./app.py /app/app.py
COPY ./requirements.txt /app/requirements.txt
COPY ./data/ /app/data/
COPY ./modules/ /app/modules/
COPY ./pages/ /app/pages/


# Installer les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# Exposer le port pour Streamlit
EXPOSE 8501

# Lancer Streamlit
CMD ["streamlit", "run", "/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]