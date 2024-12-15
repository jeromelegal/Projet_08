import requests
import json
import os
import numpy as np
import streamlit as st

API_URL = "http://api-container.germanywestcentral.azurecontainer.io:8000/predict"



def transfer_csv(url, file_path):
    # Lire le contenu du fichier depuis le chemin
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        try:
            response = requests.post(url, files=files)
            return response
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la requête vers l'API : {str(e)}")
            return None

def test_predict(uploaded_file):
    if uploaded_file is not None:
        response = transfer_csv(API_URL, uploaded_file)
        if response and response.status_code == 200:
            results = response.json()
            #predict = response.json().get("predictions")
            predict = results.get("predictions", {}).get("predictions", [])
            if not predict:
                raise ValueError("Les prédictions sont absentes du JSON")
            
            explained_value = results.get("explained_value", None)
            if explained_value is None:
                raise ValueError("La valeur expliquée (explained_value) est absente du JSON")
            
            shap_values_list = results.get("shap_values", [])
            if not shap_values_list:
                raise ValueError("Les SHAP values sont absentes ou vides")
            shap_values = np.array(shap_values_list)

            feature_names = results.get("feature_names", [])
            if not feature_names:
                raise ValueError("Les feature_names sont absentes ou vides")

        else:
            st.error("Impossible de récupérer une réponse correcte de l'API.")
    else:
        st.error("Veuillez charger un fichier CSV avant de lancer la prédiction.")
    return predict, explained_value, shap_values, feature_names