import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, NumeralTickFormatter, Div, HoverTool
from bokeh.layouts import column
import numpy as np

def create_age_distribution(df, client_age):
    # Création de l'histogramme
    hist, edges = np.histogram(df['DAYS_BIRTH'] / -365, bins=25)
    # Source de données
    source = ColumnDataSource(data={
        'left': edges[:-1],
        'right': edges[1:],
        'top': hist,
        'bottom': [0]*len(hist)
    })
    # Création du graphique
    p = figure(title="Age des clients", x_axis_label="Age (années)", y_axis_label="Quantité", plot_height=300)
    bars = p.quad(top='top', bottom='bottom', left='left', right='right',
                  source=source, fill_alpha=0.7, line_color="black")
    # Ligne pour l'âge du client
    p.line([client_age, client_age], [0, max(hist)], color='red', line_width=2, legend_label=f"Age du client: {client_age}")
    # Infobulles
    hover = HoverTool(renderers=[bars], tooltips=[
        ("Tranche d'âge", "@left - @right"),
        ("Quantité", "@top")
    ])
    p.add_tools(hover)
    p.legend.location = "top_right"
    return p

def create_income_distribution(df, client_income):
    # Création de l'histogramme
    hist, edges = np.histogram(df['AMT_INCOME_TOTAL'], bins=100)
    # Source de données
    source = ColumnDataSource(data={
        'left': edges[:-1],
        'right': edges[1:],
        'top': hist,
        'bottom': [0]*len(hist)
    })
    # Création du graphique
    p = figure(title="Revenus des clients", x_axis_label="Revenu (€)", y_axis_label="Quantité", plot_height=300)
    bars = p.quad(top='top', bottom='bottom', left='left', right='right',
                  source=source, fill_alpha=0.7, line_color="black")
    # Ligne pour le revenu du client
    p.line([client_income, client_income], [0, max(hist)], color='blue', line_width=2, legend_label=f"Revenu du client: {client_income:,.0f}")
    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    # Infobulles
    hover = HoverTool(renderers=[bars], tooltips=[
        ("Tranche de revenu", "@left{0,0} - @right{0,0}"),
        ("Quantité", "@top")
    ])
    p.add_tools(hover)
    p.legend.location = "top_right"
    return p

def create_children_distribution(df, client_children):
    global_children = df['CNT_CHILDREN'].value_counts().sort_index()
    # Source de données
    source_children = ColumnDataSource(data={
        'children': global_children.index,
        'counts': global_children.values
    })
    # Création du graphique
    p = figure(title="Nombre d'enfants par client", x_axis_label="Nombre d'enfants", y_axis_label="Quantité", plot_height=300)
    bars = p.vbar(x='children', top='counts', source=source_children, width=0.5, color="orange", legend_label="Distribution globale")
    # Ajout de la barre pour le client
    if client_children in global_children.index:
        p.vbar(x=[client_children], top=[global_children[client_children]], width=0.5, color="red", legend_label=f"Enfants du client: {client_children}")
    # Infobulles
    hover = HoverTool(renderers=[bars], tooltips=[
        ("Nombre d'enfants", "@children"),
        ("Quantité", "@counts")
    ])
    p.add_tools(hover)
    p.legend.location = "top_right"
    return p

def create_marital_status_distribution(df, client_status):
    global_status = df['NAME_FAMILY_STATUS'].value_counts()
    # Source de données
    source_status = ColumnDataSource(data={
        'status': global_status.index,
        'counts': global_status.values
    })
    # Création du graphique
    p = figure(title="Situation marital", x_axis_label="Statut marital", y_axis_label="Quantité", 
               x_range=list(global_status.index), plot_height=300)
    bars = p.vbar(x='status', top='counts', source=source_status, width=0.5, color="purple", legend_label="Distribution globale")
    # Ajout de la barre pour le client
    if client_status in global_status.index:
        p.vbar(x=[client_status], top=[global_status[client_status]], width=0.5, color="red", legend_label=f"Statut du client: {client_status}")
    # Infobulles
    hover = HoverTool(renderers=[bars], tooltips=[
        ("Statut marital", "@status"),
        ("Quantité", "@counts")
    ])
    p.add_tools(hover)
    p.xaxis.major_label_orientation = "vertical"
    p.legend.location = "top_right"
    return p

def create_credit_values_distribution(df, client_credit):
    # Création de l'histogramme
    hist, edges = np.histogram(df['AMT_CREDIT'], bins=100)
    # Source de données pour Bokeh
    source = ColumnDataSource(data={
        'left': edges[:-1],  # Bords gauches des barres
        'right': edges[1:],  # Bords droits des barres
        'top': hist,         # Hauteur des barres (quantité)
        'bottom': [0]*len(hist)  # Bas des barres
    })
    # Création de la figure
    p = figure(
        title="Quantité de crédits totale par montant",
        x_axis_label="Valeur (€)",
        y_axis_label="Quantité",
        plot_height=358
    )
    # Ajout des barres de l'histogramme
    bars = p.quad(
        top='top', bottom='bottom', left='left', right='right',
        source=source, fill_alpha=0.7, line_color="black"
    )
    # Ajout de la ligne de crédit du client
    p.line(
        [client_credit, client_credit], [0, max(hist)],
        color='blue', line_width=2, legend_label=f"Client credit: {client_credit:,.0f}"
    )
    # Formatage de l'axe X
    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    # Configuration des infobulles
    hover = HoverTool(renderers=[bars], tooltips=[
        ("Valeur", "@left{0,0} - @right{0,0}"),  # Intervalle des valeurs
        ("Quantité", "@top")                    # Quantité pour chaque barre
    ])
    # Ajout des outils
    p.add_tools(hover)
    p.legend.location = "top_right"

    return p

def create_credit_probability_dotplot(df, client_proba):
    df['status'] = np.where(df['proba'] > 0.3, 'Accordé', 'Refusé')
    df['color'] = np.where(df['proba'] > 0.3, '#377eb8', '#ff7f00')
    df['icon'] = np.where(df['proba'] > 0.3, '✅', '❌')
    df['y'] = 1  

    # Source pour Bokeh
    source = ColumnDataSource(data={
        'proba': df['proba'],
        'color': df['color'],
        'status': df['status'],
        'icon': df['icon'],
        'y': df['y']
    })
    
    # Création du dotplot
    p = figure(title="Distribution des Probabilités d'Obtention du Crédit", 
               x_axis_label="Probabilité", 
               y_axis_label="",
               plot_height=400, 
               plot_width=650,
               tools="pan,zoom_in,zoom_out,reset,save"
              )

    # Points pour tous les clients
    scatter  = p.scatter('proba', 'y', size=10, color='color', alpha=0.7, source=source)
    hover = HoverTool(renderers=[scatter], tooltips=[
        ("Statut", "@status"),
        ("Probabilité", "@proba{0.0%}"),
        ("Symbole", "@icon")
    ])
    p.add_tools(hover)
    
    # Point pour le client sélectionné
    p.scatter([client_proba], [1], size=18, color="blue", alpha=0.9, 
              legend_label="Client Sélectionné", marker="circle")
    
    p.line([0.3, 0.3], [0, 2], color="black", line_dash="dashed", line_width=2,
           legend_label="Seuil: 0.3")
    
    # Customisation
    p.xaxis.formatter = NumeralTickFormatter(format="0.0%")
    p.legend.location = "top_right"
    p.legend.label_text_font_size = "10pt"
    p.yaxis.visible = False  # Cache l'axe Y pour un effet clean
    
    # Image ou icône en fonction du statut du client
    if client_proba > 0.3:
        icon_html = '<div style="color: green; font-size: 20px;">✅ Crédit Accordé</div>'
    else:
        icon_html = '<div style="color: red; font-size: 20px;">❌ Crédit Refusé</div>'
    
    # Retourner la figure Bokeh et l'icône HTML
    return p, Div(text=icon_html, width=200, height=50)