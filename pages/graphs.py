import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, NumeralTickFormatter, Div
from bokeh.layouts import column
import numpy as np

def create_age_distribution(df, client_age):
    hist, edges = np.histogram(df['DAYS_BIRTH'] / -365, bins=25)
    p = figure(title="Distribution of Ages", x_axis_label="Age (years)", y_axis_label="Count", plot_height=300)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.7, line_color="black")
    p.line([client_age, client_age], [0, max(hist)], color='red', line_width=2, legend_label=f"Client Age: {client_age}")
    p.legend.location = "top_right"
    return p

def create_income_distribution(df, client_income):
    hist, edges = np.histogram(df['AMT_INCOME_TOTAL'], bins=100)
    p = figure(title="Distribution of Incomes", x_axis_label="Income (€)", y_axis_label="Count", plot_height=300) #x_axis_type="log"
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.7, line_color="black")
    p.line([client_income, client_income], [0, max(hist)], color='blue', line_width=2, legend_label=f"Client Income: {client_income:,.0f}")
    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p.legend.location = "top_right"
    return p

def create_children_distribution(df, client_children):
    global_children = df['CNT_CHILDREN'].value_counts().sort_index()
    source_children = ColumnDataSource(data=dict(children=global_children.index, counts=global_children.values))
    p = figure(title="Number of Children", x_axis_label="Number of Children", y_axis_label="Count", plot_height=300)
    p.vbar(x='children', top='counts', source=source_children, width=0.5, color="orange", legend_label="Global Distribution")
    if client_children in global_children.index:
        p.vbar(x=[client_children], top=[global_children[client_children]], width=0.5, color="red", legend_label=f"Client Children: {client_children}")
    p.legend.location = "top_right"
    return p

def create_marital_status_distribution(df, client_status):
    global_status = df['NAME_FAMILY_STATUS'].value_counts()
    source_status = ColumnDataSource(data=dict(status=global_status.index, counts=global_status.values))
    p = figure(title="Marital Status", x_axis_label="Marital Status", y_axis_label="Count", x_range=list(global_status.index), plot_height=300)
    p.vbar(x='status', top='counts', source=source_status, width=0.5, color="purple", legend_label="Global Distribution")
    if client_status in global_status.index:
        p.vbar(x=[client_status], top=[global_status[client_status]], width=0.5, color="red", legend_label=f"Client Status: {client_status}")
    p.xaxis.major_label_orientation = "vertical"
    p.legend.location = "top_right"
    return p

def create_credit_values_distribution(df, client_credit):
    hist, edges = np.histogram(df['AMT_CREDIT'], bins=100)
    p = figure(title="Distribution of Credits", x_axis_label="Value (€)", y_axis_label="Count", plot_height=358) #x_axis_type="log"
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.7, line_color="black")
    p.line([client_credit, client_credit], [0, max(hist)], color='blue', line_width=2, legend_label=f"Client credit: {client_credit:,.0f}")
    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p.legend.location = "top_right"
    return p

def create_credit_probability_dotplot(df, client_proba):
    df['status'] = np.where(df['proba'] > 0.3, 'Accordé', 'Refusé')
    df['color'] = np.where(df['proba'] > 0.3, 'green', 'red')
    df['y'] = 1  

    # Source pour Bokeh
    source = ColumnDataSource(data={
        'proba': df['proba'],
        'color': df['color'],
        'y': df['y']
    })
    
    # Création du dotplot
    p = figure(title="Distribution des Probabilités d'Obtention du Crédit", 
               x_axis_label="Probabilité", y_axis_label="",
               plot_height=400, plot_width=650)

    # Points pour tous les clients
    p.scatter('proba', 'y', size=8, color='color', alpha=0.6, source=source)
    
    # Point pour le client sélectionné
    p.scatter([client_proba], [1], size=12, color="blue", legend_label="Client Sélectionné")
    p.line([0.3, 0.3], [0, 2], color="black", line_dash="dashed", legend_label="Seuil: 0.3")
    
    # Customisation
    p.xaxis.formatter = NumeralTickFormatter(format="0.0%")
    p.legend.location = "top_right"
    p.yaxis.visible = False  # Cache l'axe Y pour un effet clean
    
    # Image ou icône en fonction du statut du client
    if client_proba > 0.3:
        icon_html = '<div style="color: green; font-size: 20px;">✅ Crédit Accordé</div>'
    else:
        icon_html = '<div style="color: red; font-size: 20px;">❌ Crédit Refusé</div>'
    
    # Retourner la figure Bokeh et l'icône HTML
    return p, Div(text=icon_html, width=200, height=50)