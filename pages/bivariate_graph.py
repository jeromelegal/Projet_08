from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
import numpy as np
from bokeh.models import Whisker
import streamlit as st
import pandas as pd

def auto_graph_type(x, y, data):
    """
    Détermine automatiquement le type de graphique en fonction des types de variables.
    """
    if pd.api.types.is_numeric_dtype(data[x]) and pd.api.types.is_numeric_dtype(data[y]):
        return "Scatter Plot"
    elif pd.api.types.is_numeric_dtype(data[x]) or pd.api.types.is_numeric_dtype(data[y]):
        return "Box Plot"
    else:
        return "Bar Plot"

def create_bivariate_graph(data):
    """
    Fonction qui génère un graphique Bokeh bivarié interactif
    en fonction des choix de l'utilisateur via un formulaire.
    """
    # Sélection du type de graphique et des variables
    with st.form("bivariate_form"):
        st.write("### Sélectionnez les paramètres du graphique")
        x_var = st.selectbox("1ère feature (X)", data.columns, index=0)
        y_var = st.selectbox("2ème feature (Y)", data.columns, index=1)
        submitted = st.form_submit_button("Afficher le Graphique")

    # Création du graphique après validation
    if submitted:
        graph_type = auto_graph_type(x_var, y_var, data)

        if graph_type == "Scatter Plot":
            source = ColumnDataSource(data={x_var: data[x_var], y_var: data[y_var]})
            p = figure(title=f"Scatter Plot : {x_var} vs {y_var}",
                       x_axis_label=x_var, y_axis_label=y_var,
                       plot_width=700, plot_height=500)
            p.circle(x=x_var, y=y_var, source=source, size=8, alpha=0.6, color="navy")
            hover = HoverTool(tooltips=[(x_var, f"@{x_var}"), (y_var, f"@{y_var}")])
            p.add_tools(hover)

        elif graph_type == "Box Plot":
            # Box Plot : Calcul des statistiques pour chaque groupe
            grouped = data.groupby(x_var)[y_var].describe()
            grouped = grouped.reset_index()

            # Récupération des valeurs pour le Box Plot
            q1 = grouped[("50%")] - (grouped[("75%")] - grouped[("50%")])  # Q1 approximation
            q2 = grouped[("50%")]  # Médiane
            q3 = grouped[("75%")]  # Q3
            lower = grouped[("min")]
            upper = grouped[("max")]

            # Création d'une source Bokeh
            source = ColumnDataSource(data=dict(
                categories=grouped[x_var].astype(str),
                lower=lower,
                q1=q1,
                median=q2,
                q3=q3,
                upper=upper
            ))

            # Initialisation du graphique
            p = figure(title=f"Box Plot : {x_var} vs {y_var}",
                    x_range=grouped[x_var].astype(str).tolist(),
                    x_axis_label=x_var, y_axis_label=y_var,
                    plot_width=700, plot_height=500)

            # Création des boîtes
            p.vbar(x="categories", top="q3", bottom="q1", width=0.7, fill_color="lightblue", source=source)

            # Ajout des "moustaches" (Whiskers)
            whisker = Whisker(base="categories", upper="upper", lower="lower", source=source, level="annotation")
            whisker.upper_head.size = 10
            whisker.lower_head.size = 10
            p.add_layout(whisker)

            # Ajout de la médiane
            p.segment(x0="categories", y0="median", x1="categories", y1="median", line_width=2, color="navy", source=source)

        elif graph_type == "Bar Plot":
            # Pour deux variables qualitatives
            counts = data.groupby([x_var, y_var]).size().reset_index(name='count')
            source = ColumnDataSource(counts)
            p = figure(title=f"Bar Plot : {x_var} vs {y_var}",
                       x_axis_label=x_var, y_axis_label="Count",
                       x_range=counts[x_var].astype(str).unique().tolist(),
                       plot_width=700, plot_height=500)
            p.vbar(x=x_var, top='count', width=0.7, source=source)

        else:
            st.warning("Impossible de générer un graphique. Sélectionnez au moins une variable quantitative.")
            return None

        st.info(f"Type de graphique détecté automatiquement : **{graph_type}**")
        return p
