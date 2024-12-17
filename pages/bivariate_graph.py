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

from bokeh.models import ColumnDataSource, HoverTool, Whisker
from bokeh.plotting import figure
import pandas as pd
import streamlit as st

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
    # Sélection des variables
    with st.form("bivariate_form"):
        st.write("### Sélectionnez les paramètres du graphique")
        x_var = st.selectbox("1ère feature (X)", data.columns, index=0)
        y_var = st.selectbox("2ème feature (Y)", data.columns, index=1)
        submitted = st.form_submit_button("Afficher le Graphique")

    if submitted:
        graph_type = auto_graph_type(x_var, y_var, data)

        # SCATTER PLOT
        if graph_type == "Scatter Plot":
            source = ColumnDataSource(data={x_var: data[x_var], y_var: data[y_var]})
            p = figure(title=f"Scatter Plot : {x_var} vs {y_var}",
                       x_axis_label=x_var, y_axis_label=y_var,
                       plot_width=700, plot_height=500)
            p.circle(x=x_var, y=y_var, source=source, size=8, alpha=0.6, color="navy")

            # Infobulles
            hover = HoverTool(tooltips=[(x_var, f"@{x_var}"), (y_var, f"@{y_var}")])
            p.add_tools(hover)

        # BOX PLOT
        elif graph_type == "Box Plot":
            if not pd.api.types.is_numeric_dtype(data[y_var]):
                st.warning(f"{y_var} n'est pas numérique. Impossible de générer un Box Plot.")
                return None

            grouped = data.groupby(x_var)[y_var].describe()
            grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
            grouped = grouped.reset_index()

            lower = grouped[f"{y_var}_min"]
            q1 = grouped[f"{y_var}_25%"]
            median = grouped[f"{y_var}_50%"]
            q3 = grouped[f"{y_var}_75%"]
            upper = grouped[f"{y_var}_max"]

            source = ColumnDataSource(data=dict(
                categories=grouped[x_var].astype(str),
                lower=lower, q1=q1, median=median, q3=q3, upper=upper
            ))

            p = figure(title=f"Box Plot : {x_var} vs {y_var}",
                       x_range=grouped[x_var].astype(str).tolist(),
                       plot_width=700, plot_height=500)

            # Boîtes
            p.vbar(x="categories", top="q3", bottom="q1", width=0.7, fill_color="lightblue", source=source)

            # Moustaches
            whisker = Whisker(base="categories", upper="upper", lower="lower", source=source)
            p.add_layout(whisker)

            # Médiane
            p.segment(x0="categories", y0="median", x1="categories", y1="median", line_width=2, color="navy", source=source)

            # Infobulles
            hover = HoverTool(tooltips=[
                ("Category", "@categories"),
                ("Q1", "@q1"),
                ("Median", "@median"),
                ("Q3", "@q3"),
                ("Min", "@lower"),
                ("Max", "@upper")
            ])
            p.add_tools(hover)

        # BAR PLOT
        elif graph_type == "Bar Plot":
            counts = data.groupby([x_var, y_var]).size().reset_index(name='count')
            source = ColumnDataSource(counts)

            p = figure(title=f"Bar Plot : {x_var} vs {y_var}",
                       x_axis_label=x_var, y_axis_label="Count",
                       x_range=counts[x_var].astype(str).unique().tolist(),
                       plot_width=700, plot_height=500)
            p.vbar(x=x_var, top='count', width=0.7, source=source, color="orange")

            # Infobulles
            hover = HoverTool(tooltips=[
                (x_var, f"@{x_var}"),
                (y_var, f"@{y_var}"),
                ("Count", "@count")
            ])
            p.add_tools(hover)

        else:
            st.warning("Impossible de générer un graphique. Sélectionnez au moins une variable quantitative.")
            return None

        st.info(f"Type de graphique détecté automatiquement : **{graph_type}**")
        return p
