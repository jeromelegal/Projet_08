import streamlit as st
import pandas as pd
from pages.predictions import test_predict
from pages.graphs import (
    create_age_distribution, create_income_distribution,
    create_children_distribution, create_marital_status_distribution,
    create_credit_values_distribution, create_credit_probability_dotplot
)
from pages.shap_explenations import compare_feature_importance_bokeh
from pages.bivariate_graph import create_bivariate_graph
import tempfile
import pickle

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    div.streamlit-expanderContent { 
        height: 350px; /* Ajustement des graphiques dans les onglets */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        /* Cache le menu des pages généré automatiquement */
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#################################################################################
# prepare data

data_path = "data/full_test_prediction.csv"

@st.cache_data
def load_data():
    return pd.read_csv(data_path)

@st.cache_data
def load_bivariate_data():
    return pd.read_csv("data/bivariate_analysis_dataset.csv")

clients = load_data()
clients_bivariate = load_bivariate_data()

# retrieve global shap values
with open("./data/global_shap_values.pkl", "rb") as file:
    global_shap_values = pickle.load(file)

#################################################################################
# main title
st.title("Tableau de Bord")


# client selection on selectbox
st.sidebar.title("Sélectionnez un client")
client_id = st.sidebar.selectbox("ID Client", clients["SK_ID_CURR"])
client_data = clients[clients["SK_ID_CURR"] == client_id]

with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w') as temp_file:
    client_data.to_csv(temp_file.name, index=False)
    temp_file_path = temp_file.name  # Récupérer le chemin du fichier temporaire
predict, explained_value, local_shap_values, feature_names = test_predict(temp_file_path)

#################################################################################
# client variables

client_age = round(client_data['DAYS_BIRTH'].values[0] / -365, 0)
client_income = client_data['AMT_INCOME_TOTAL'].values[0]
client_children = client_data['CNT_CHILDREN'].values[0]
client_status = client_data['NAME_FAMILY_STATUS'].values[0]
client_credit = client_data['AMT_CREDIT'].values[0]

#################################################################################
# graphs

tab1_graph = create_age_distribution(clients, client_age)
tab2_graph = create_income_distribution(clients, client_income)
tab3_graph = create_children_distribution(clients, client_children)
tab4_graph = create_marital_status_distribution(clients, client_status)
credit_graph = create_credit_values_distribution(clients, client_credit)
dotplot, decision_div = create_credit_probability_dotplot(clients, client_data['proba'].values[0])

#################################################################################
# sidebar

st.sidebar.markdown("---")
st.sidebar.subheader("Informations sur le client")
st.sidebar.write(f"**ID :** {client_data['SK_ID_CURR'].values[0]}")
st.sidebar.write(f"**Sexe :** {'Femme' if client_data['CODE_GENDER'].values[0] == 'F' else 'Homme'}")
st.sidebar.write(f"**Revenu total :** {client_data['AMT_INCOME_TOTAL'].values[0]:,.0f}")
st.sidebar.write(f"**Age :** {round(client_data['DAYS_BIRTH'].values[0] / -365, 0)} ans")
st.sidebar.write(f"**Statut familial :** {client_data['NAME_FAMILY_STATUS'].values[0]}")
st.sidebar.write(f"**Nombre d'enfants :** {client_data['CNT_CHILDREN'].values[0]}")
st.sidebar.write(f"**Possède une voiture :** {'Oui' if client_data['FLAG_OWN_CAR'].values[0] == 'Y' else 'Non'}")
st.sidebar.write(f"**Possède un logement :** {'Oui' if client_data['FLAG_OWN_REALTY'].values[0] == 'Y' else 'Non'}")
st.sidebar.markdown("---")
st.sidebar.subheader("Informations sur le crédit")
st.sidebar.write(f"**Type de contrat :** {client_data['NAME_CONTRACT_TYPE'].values[0]}")
st.sidebar.write(f"**Montant :** {client_data['AMT_CREDIT'].values[0]:,.0f}")
st.sidebar.markdown(f"**Statut :** {decision_div.text}", unsafe_allow_html=True)
st.sidebar.write(f"**Probabilité :** {client_data['proba'].values[0]:.4f}")

#################################################################################
# Layout 

col_left, col_right = st.columns([3, 2])

# left area : 4 graphs (2x2)
with col_left:
    left_col1, left_col2 = st.columns(2)

    with left_col1:
        with st.container(border=True):
            st.subheader("Probabilité d'Obtention du Crédit")
            st.bokeh_chart(dotplot, use_container_width=True)
            st.caption(
        "Ce graphique montre la probabilité estimée que le crédit soit accordé. "
        "Le point bleu représente le client sélectionné, et la ligne en pointillés indique le seuil de refus (0.3)."
        " "
    )

        with st.container(border=True):
            st.subheader("Graphiques Clients")
            tab = st.tabs([
                "Distribution de l'âge", 
                "Distribution des revenus", 
                "Nombre d'enfant", 
                "Statut marital"
            ])

            with tab[0]:
                st.bokeh_chart(tab1_graph)
                st.caption("Ce graphique montre la distribution des âges dans l'ensemble des clients, avec une ligne rouge indiquant l'âge du client sélectionné.")
    
            with tab[1]:
                st.bokeh_chart(tab2_graph)
                st.caption("Ce graphique représente la distribution des revenus des clients, avec une ligne bleue indiquant le revenu du client sélectionné.")
    
            with tab[2]:
                st.bokeh_chart(tab3_graph)
                st.caption("Ce graphique indique le nombre d'enfants pour les clients, en comparaison avec le nombre d'enfants du client sélectionné (barre rouge).")
    
            with tab[3]:
                st.bokeh_chart(tab4_graph)
                st.caption("Ce graphique montre le statut marital des clients, en comparant avec le statut marital du client sélectionné (barre rouge).")

    with left_col2:
        with st.container(border=True):
            st.subheader("Caractéristiques importantes")
            fig_shap = compare_feature_importance_bokeh(global_shap_values, local_shap_values, feature_names)
            st.bokeh_chart(fig_shap, use_container_width=True)
            st.caption(
                "Ce graphique représente les 10 caractéristiques qui ont le plus influencées la décision du modèle (classement d'importance du haut vers le bas)."
                "En rouge nous retrouvons la valeur pour le client, en bleu la valeur moyenne."
            )

        with st.container(border=True):
            st.subheader("Distribution des Crédits")
            st.bokeh_chart(credit_graph, use_container_width=True)
            st.caption(
        "Ce graphique représente la distribution de toutes les demandes de crédits. "
        " "
    )

# right area : bivariate graph
with col_right:
    with st.container(border=True):
        st.subheader("Analyse de la Relation entre Deux Variables")
        bivariate_graph = create_bivariate_graph(clients_bivariate)
        if bivariate_graph:
            st.bokeh_chart(bivariate_graph, use_container_width=True)
        st.caption(
            "Ce graphique permet d’analyser la relation entre la variable X et Y. Utilisez le zoom pour explorer les détails."          
        )
        st.caption("🔍 **Astuce** : Vous pouvez zoomer ou survoler les points pour voir les détails.")


