import shap
import matplotlib.pyplot as plt

from bokeh.models import ColumnDataSource, Legend, HoverTool
from bokeh.plotting import figure
from bokeh.transform import dodge
import pandas as pd
import numpy as np


def shap_bar_plot(shap_values, feature_names):
    # Extraire les valeurs SHAP pour l'instance sélectionnée
    shap_values_instance = shap.Explanation(
        values=shap_values[0],  
        base_values=None,                   
        data=None,
        feature_names=feature_names                           
    )
    # Créer le graphique SHAP statique
    fig, ax = plt.subplots(figsize=(12, 6))
    shap.plots.bar(shap_values_instance, show=False)  
    return fig

def compare_feature_importance_bokeh(global_shap_values, local_shap_values, feature_names):
    """
    Compare les SHAP values locales et globales avec un barplot Bokeh.
    - global_shap_values : shap.Explanation (sortie de l'explainer SHAP).
    - local_shap_values : Liste des valeurs SHAP pour un client spécifique.
    - feature_names : Liste des noms des features.
    """
    # Étape 1 : Calcul des SHAP values globales (moyenne absolue)
    global_shap_mean = np.abs(global_shap_values.values).mean(axis=0)

    # Étape 2 : S'assurer que local_shap_values est 1D
    local_shap_values = np.squeeze(local_shap_values)

    # Étape 3 : Créer un DataFrame pour les valeurs locales et globales
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Global Importance": global_shap_mean,
        "Local Importance": local_shap_values
    })

    # Trier par importance locale et sélectionner les 10 features principales
    shap_df = shap_df.sort_values(by="Local Importance", ascending=False).head(10)

    # Inverser l'ordre pour avoir les features les plus importantes en haut
    shap_df = shap_df.iloc[::-1]

    # Source pour Bokeh
    source = ColumnDataSource(data={
        "Feature": shap_df["Feature"],
        "Global": shap_df["Global Importance"],
        "Local": shap_df["Local Importance"]
    })

    # Étape 4 : Création du barplot avec Bokeh
    p = figure(
        y_range=list(shap_df["Feature"]),  # Range inversé
        title="Comparaison Feature Importance (Top 10)",
        plot_height=400,  # Hauteur ajustée
        plot_width=650,   # Largeur ajustée pour éviter les débordements
        toolbar_location=None,
        tools=""
    )

    # Ajouter les barres pour les valeurs globales et locales
    p.hbar(y=dodge("Feature", -0.15, range=p.y_range), right="Global", height=0.3,
           color="skyblue", source=source, legend_label="Globale")
    p.hbar(y=dodge("Feature", 0.15, range=p.y_range), right="Local", height=0.3,
           color="red", source=source, legend_label="Locale")

    # Ajouter un HoverTool pour afficher les valeurs au survol
    hover = HoverTool(tooltips=[
        ("Feature", "@Feature"),
        ("Importance Globale", "@Global{0.000}"),
        ("Importance Locale", "@Local{0.000}")
    ])
    p.add_tools(hover)

    # Personnalisation
    p.legend.location = "top_right"
    p.xaxis.axis_label = "SHAP Value (Importance)"
    p.ygrid.grid_line_color = None
    p.outline_line_color = None

    return p