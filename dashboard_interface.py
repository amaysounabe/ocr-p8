import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go
import plotly.express as px
import sklearn
import math
import io
import streamlit as st
import shap
from dashboard_fonctions import *

# importation du modèle
model_path = "./data/xgbclassifier.pkl"
model = load(model_path)

# importation de l'explainer
explainer_path = "./data/shap_explainer.pkl"
explainer = load(explainer_path)

# liste des colonnes
column_names_path = './data/column_names.pkl'
column_names = load(column_names_path)

# importation des données
if 'data' not in st.session_state:
    st.session_state.data = pd.read_csv('./data/df_test.csv')

data = st.session_state.data

# tri des IDs de clients
sorted_ids = sorted(data['SK_ID_CURR'].unique())
    
# liste des 10 features les plus importantes
columns_to_display = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'PAYMENT_RATE', 'EXT_SOURCE_1', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED', 'AMT_ANNUITY', 'AMT_CREDIT', 'CODE_GENDER_F']

# importation des styles
with open("./styles/dashboard_style.css") as f:
    css = f.read()

css = f"{css}"
st.markdown(f"<style>{css}</style>", unsafe_allow_html = True)

st.markdown("""<div class="title">Simulation de Prêt Client</div>""", unsafe_allow_html=True)

# Utilisation de CSS pour appliquer un border-radius à l'image
st.markdown("""
    <img class="styled-image" src="https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png" alt="Banner Image">
""", unsafe_allow_html=True)


# Initialisation de st.session_state
if 'current_id_index' not in st.session_state:
    st.session_state.current_id_index = 0  # Pas de sélection initiale
if 'age' not in st.session_state:
    st.session_state.age = 30
if 'nb_children' not in st.session_state:
    st.session_state.nb_children = 0
if 'income' not in st.session_state:
    st.session_state.income = 50
if 'tenure' not in st.session_state:
    st.session_state.tenure = 10
if 'good_price' not in st.session_state:
    st.session_state.good_price = 20
if 'rate' not in st.session_state:
    st.session_state.rate = 5

# Affichage de l'ID client actuel
current_client_id = sorted_ids[st.session_state.current_id_index]

# Ajouter une option vide à la liste
options = ['Sélectionner un client'] + [int(id) for id in sorted_ids]

selected_client_id = st.selectbox(
    'Sélectionnez l\'identifiant client :',
    options = sorted_ids,
    index = st.session_state.current_id_index,
    format_func = lambda x:f"Client {int(x)}"
)

# Mise à jour de l'index actuel si l'utilisateur sélectionne un ID dans le selectbox
if selected_client_id != current_client_id:
    st.session_state.current_id_index = sorted_ids.index(selected_client_id)
    st.session_state.current_client_id = current_client_id
    current_client_id = selected_client_id
    


st.markdown(f"""<div class="client-id">ID du client actuel : {int(current_client_id)}</div>""", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"""<div class="subsubtitle">Informations clients</div>""", unsafe_allow_html = True)
client_info_table = get_client_infos(current_client_id, data)
        
if not client_info_table.empty:
    st.session_state.age = client_info_table[client_info_table['Caractéristiques'] == 'Âge']['Données'].values[0]
    st.session_state.nb_children = client_info_table[client_info_table['Caractéristiques'] == 'Nombre d\'enfants']['Données'].values[0]
    st.session_state.income = client_info_table[client_info_table['Caractéristiques'] == 'Revenus totaux']['Données'].values[0]
    st.session_state.tenure = client_info_table[client_info_table['Caractéristiques'] == 'Ancienneté']['Données'].values[0]
    st.session_state.good_price = client_info_table[client_info_table['Caractéristiques'] == 'Prix du bien']['Données'].values[0]
    st.session_state.rate = client_info_table[client_info_table['Caractéristiques'] == 'Taux de remboursement']['Données'].values[0]
    
    # Créer un DataFrame avec uniquement les valeurs et caractéristiques
    client_info = pd.DataFrame({
        "Caractéristiques": ["Âge", "Nombre d'enfants", "Revenus totaux (en k€)", "Années travaillées", "Prix du bien", "Taux de remboursement"],
        "Valeurs": [str(st.session_state.age),
                    str(st.session_state.nb_children),
                    str(st.session_state.income),
                    str(st.session_state.tenure),
                    str(st.session_state.good_price),
                    str(st.session_state.rate)
                   ]
    })
    st.dataframe(client_info.set_index("Caractéristiques").T.set_index("Âge"), use_container_width = True)# Affichage avec st.table() sans afficher les noms de colonnes
    
st.markdown("---")  
st.markdown(f"""<div class="subsubtitle">Modification des informations client</div>""", unsafe_allow_html = True)
    
st.session_state.age = st.slider("Âge", min_value=18, max_value=100, value=st.session_state.age, step = 1)
st.session_state.nb_children = st.slider("Nombre d'enfants", min_value=0, max_value=20, value=st.session_state.nb_children, step = 1)
st.session_state.income = st.slider("Revenus totaux (k€)", min_value=0, max_value=500, value=st.session_state.income, step = 1)
st.session_state.tenure = st.slider("Années travaillées", min_value=0, max_value=st.session_state.age, value=st.session_state.tenure, step = 1)
st.session_state.good_price = st.slider("Prix du bien (k€)", min_value=0, max_value=1000, value=st.session_state.good_price, step = 1)
st.session_state.rate = st.slider("Taux de remboursement (%)", min_value=0, max_value=30, value=st.session_state.rate, step = 1)
            
if st.button("Mettre à jour les informations du client"):
    # Mise à jour des valeurs dans le DataFrame
    data.loc[data['SK_ID_CURR'] == current_client_id, 'DAYS_BIRTH'] = 365 * st.session_state.age
    data.loc[data['SK_ID_CURR'] == current_client_id, 'CNT_CHILDREN'] = st.session_state.nb_children
    data.loc[data['SK_ID_CURR'] == current_client_id, 'AMT_INCOME_TOTAL'] = 1000 * st.session_state.income
    data.loc[data['SK_ID_CURR'] == current_client_id, 'DAYS_EMPLOYED'] = 365 * st.session_state.tenure
    data.loc[data['SK_ID_CURR'] == current_client_id, 'AMT_GOODS_PRICE'] = 1000 * st.session_state.good_price
    data.loc[data['SK_ID_CURR'] == current_client_id, 'PAYMENT_RATE'] = 0.01 * st.session_state.rate
    st.write("Informations du client mises à jour")
    st.write(data.loc[data['SK_ID_CURR'] == current_client_id, ['DAYS_BIRTH', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED']])
    
st.markdown("---")

# Nuages de points
st.markdown("""<div class="subtitle">Nuage de points</div>""", unsafe_allow_html=True)

columns_to_display_mapping = {col.replace('_', ' '): col for col in columns_to_display}

x_axis_display = st.selectbox("Sélectionnez la variable à positionner en abscisse", list(columns_to_display_mapping.keys()))
y_axis_display = st.selectbox("Sélectionnez la variable à positionner en ordonnées", list(columns_to_display_mapping.keys()))

if st.button("Affichage du graphique"):
    x_axis = columns_to_display_mapping[x_axis_display]
    y_axis = columns_to_display_mapping[y_axis_display]
    
    fig = scatter_plot(data, x_axis, y_axis, current_client_id)
    st.plotly_chart(fig)

st.markdown("---")
st.markdown("""<div class="subtitle">Prédiction</div>""", unsafe_allow_html=True)

# Détails des prédictions avec SHAP values
if st.button("Affichage de l'explication"):
    data_shap = get_client_shap_values(current_client_id, data, model, explainer, column_names, columns_to_display).T
    data_shap.columns = ['SHAP Value']
    data_shap.reset_index(inplace=True)
    data_shap.rename(columns={'index': 'Feature'}, inplace=True)
    fig = plot_shap_values(data_shap, current_client_id)
    st.plotly_chart(fig)
    
    

# Prédiction (proba + décision)
if st.button("Simulation"):
    client_data = get_client_data(current_client_id, data, column_names)
    optimal_threshold = np.load('./data/optimal_threshold.npy').item()
    
    if client_data is not None:
        predictions = predict(client_data, model)
        if predictions is not None:
            prediction_value = predictions[0, 1]
            prediction_value_percent = 100 * prediction_value

            # Créer la jauge avec Plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_value_percent,
                number = {'suffix': "%"},  # Ajoute le symbole % à la valeur
                gauge = {
                    'shape': "angular",  # Forme angulaire pour une jauge semi-circulaire
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black", 'showticklabels': True},
                    'bar': {'color': "green" if prediction_value <= optimal_threshold else "red"},  # Couleur conditionnelle
                    'steps': [
                        {'range': [0, optimal_threshold * 100], 'color': "lightgreen"},
                        {'range': [optimal_threshold * 100, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': optimal_threshold * 100 # Afficher le seuil optimal
                    }
                },
                domain = {'x': [0, 1], 'y': [0, 1]},
                #title = {'text': "Probabilité de Prêt Accordé"}
            ))

            # Afficher la jauge dans Streamlit
            st.plotly_chart(fig)
            
            if prediction_value <= optimal_threshold:
                st.markdown("<h1 style='color:green; text-align:center;'>PRÊT ACCORDÉ</h1>", unsafe_allow_html=True)
            else:
                st.markdown("<h1 style='color:red; text-align:center;'>PRÊT REFUSÉ</h1>", unsafe_allow_html=True)
        
    else:
        st.error("ID client non trouvé dans les données.")
