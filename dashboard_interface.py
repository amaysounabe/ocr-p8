import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go
import plotly.express as px
from dashboard_fonctions import *

# Importation des modèles et données
model_path = "./data/xgbclassifier.pkl"
model = load(model_path)

explainer_path = "./data/shap_explainer.pkl"
explainer = load(explainer_path)

column_names_path = './data/column_names.pkl'
column_names = load(column_names_path)

if 'data' not in st.session_state:
    st.session_state.data = pd.read_csv('./data/df_test.csv')

data = st.session_state.data
sorted_ids = sorted(data['SK_ID_CURR'].unique())
columns_to_display = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'PAYMENT_RATE', 'EXT_SOURCE_1', 'AMT_GOODS_PRICE', 
                      'DAYS_EMPLOYED', 'AMT_ANNUITY', 'AMT_CREDIT', 'CODE_GENDER_F']

# Initialisation de st.session_state
if 'selected_client_id' not in st.session_state:
    st.session_state.selected_client_id = sorted_ids[0]  # Pas de sélection initiale
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
    
# importation des styles
with open("./styles/dashboard_style.css") as f:
    css = f.read()

css = f"{css}"
st.markdown(f"<style>{css}</style>", unsafe_allow_html = True)

st.markdown("""<div class="title">Simulation de Prêt Client</div>""", unsafe_allow_html=True)
st.markdown("""
    <img class="styled-image" src="https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png" alt="Banner Image">
""", unsafe_allow_html=True)

# Créer le menu dans la barre latérale
with st.sidebar:
    st.markdown(f"""<div class="subsubtitle">Menu principal</div>""", unsafe_allow_html=True)
    menu = option_menu(
        "", 
        ["Informations clients", "Analyse graphique", "Analyse prédictive"], 
        icons=['person-fill', 'bar-chart-fill', 'activity'],  # Choisissez les icônes appropriées
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"background-color": "transparent", "padding": "5!important"},
            "icon": {"color": "green", "font-size": "25px"},
            "nav-link": {"font-family": "serif", "font-size": "18px", "text-align": "left", "margin": "5px"},
            "nav-link-selected": {"background-color": "black"},  # Couleur de l'option sélectionnée
        }
    )
st.sidebar.markdown("---")
# Sélection d'un client dans la sidebar
st.sidebar.markdown(f"""<div class="subsubtitle">Sélection du client</div>""", unsafe_allow_html=True)
selected_client_id = st.sidebar.selectbox(
    "",
    options=sorted_ids,
    format_func=lambda x: f"Client {int(x)}",
    index=sorted_ids.index(st.session_state.selected_client_id)
)


# Partie 1 : Informations clients
if menu == "Informations clients":
    st.markdown(f"""<div class="client-id">ID du client actuel : {int(selected_client_id)}</div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="subsubtitle">Informations clients</div>""", unsafe_allow_html=True)
    
    client_info_table = get_client_infos(selected_client_id, data)
    
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
        data.loc[data['SK_ID_CURR'] == selected_client_id, 'DAYS_BIRTH'] = 365 * st.session_state.age
        data.loc[data['SK_ID_CURR'] == selected_client_id, 'CNT_CHILDREN'] = st.session_state.nb_children
        data.loc[data['SK_ID_CURR'] == selected_client_id, 'AMT_INCOME_TOTAL'] = 1000 * st.session_state.income
        data.loc[data['SK_ID_CURR'] == selected_client_id, 'DAYS_EMPLOYED'] = 365 * st.session_state.tenure
        data.loc[data['SK_ID_CURR'] == selected_client_id, 'AMT_GOODS_PRICE'] = 1000 * st.session_state.good_price
        data.loc[data['SK_ID_CURR'] == selected_client_id, 'PAYMENT_RATE'] = 0.01 * st.session_state.rate
        
        st.success("Informations du client mises à jour avec succès.")
        
    st.markdown("---")

# Partie 2 : Graphiques (Nuage de points)
elif menu == "Analyse graphique":
    st.markdown(f"""<div class="client-id">ID du client actuel : {int(selected_client_id)}</div>""", unsafe_allow_html=True)
    st.markdown("""<div class="subtitle">Nuage de points</div>""", unsafe_allow_html=True)
    
    columns_to_display_mapping = {col.replace('_', ' '): col for col in columns_to_display}
    
    x_axis_display = st.selectbox("Sélectionnez la variable à positionner en abscisse", list(columns_to_display_mapping.keys()))
    y_axis_display = st.selectbox("Sélectionnez la variable à positionner en ordonnées", list(columns_to_display_mapping.keys()))
    
    if st.button("Affichage du graphique"):
        x_axis = columns_to_display_mapping[x_axis_display]
        y_axis = columns_to_display_mapping[y_axis_display]
        
        fig = scatter_plot(data, x_axis, y_axis, selected_client_id)
        st.plotly_chart(fig)

# Partie 3 : Prédiction
elif menu == "Analyse prédictive":
    st.markdown(f"""<div class="client-id">ID du client actuel : {int(selected_client_id)}</div>""", unsafe_allow_html=True)
    st.markdown("""<div class="subtitle">Prédiction</div>""", unsafe_allow_html=True)
    
    if st.button("Affichage de l'explication"):
        data_shap = get_client_shap_values(selected_client_id, data, model, explainer, column_names, columns_to_display).T
        data_shap.columns = ['SHAP Value']
        data_shap.reset_index(inplace=True)
        data_shap.rename(columns={'index': 'Feature'}, inplace=True)
        fig = plot_shap_values(data_shap, selected_client_id)
        st.plotly_chart(fig)
    
    if st.button("Simulation"):
        client_data = get_client_data(selected_client_id, data, column_names)
        optimal_threshold = np.load('./data/optimal_threshold.npy').item()
        
        if client_data is not None:
            predictions = predict(client_data, model)
            if predictions is not None:
                prediction_value = predictions[0, 1]
                prediction_value_percent = 100 * prediction_value
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_value_percent,
                    number={'suffix': "%"},
                    gauge={
                        'shape': "angular",
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
                        'bar': {'color': "green" if prediction_value <= optimal_threshold else "red"},
                        'steps': [
                            {'range': [0, optimal_threshold * 100], 'color': "lightgreen"},
                            {'range': [optimal_threshold * 100, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'value': optimal_threshold * 100}
                    }
                ))
                st.plotly_chart(fig)
                
                if prediction_value <= optimal_threshold:
                    st.markdown("<h1 style='color:green; text-align:center;'>PRÊT ACCORDÉ</h1>", unsafe_allow_html=True)
                else:
                    st.markdown("<h1 style='color:red; text-align:center;'>PRÊT REFUSÉ</h1>", unsafe_allow_html=True)
