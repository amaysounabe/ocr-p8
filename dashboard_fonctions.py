# Fonctions utilisées pour le dashboard Streamlit
import numpy as np
import pandas as pd
import math
import sklearn
import plotly.graph_objects as go
import shap

# fonction pour obtenir les données du client en fonction de l'ID
def get_client_data(current_client_id, data, column_names):
    client_data = data[data['SK_ID_CURR'] == current_client_id]
    if client_data.empty:
        return None
    # Retourne les features du client (sans l'ID)
    return client_data[column_names].values.tolist()


# fonction pour récupérer les infos du client
def get_client_infos(current_client_id, data):
    client_data = data[data['SK_ID_CURR'] == current_client_id]
    if not client_data.empty:
        age_client = math.floor(client_data['DAYS_BIRTH'].values[0] / 365)
        nb_children_client = round(client_data['CNT_CHILDREN'].values[0])
        income_client = round(client_data['AMT_INCOME_TOTAL'].values[0] / 1000)
        tenure_client = math.floor(client_data['DAYS_EMPLOYED'].values[0] / 365)
        goodprice_client = round(client_data['AMT_GOODS_PRICE'].values[0] / 1000)
        rate_client = round(client_data['PAYMENT_RATE'].values[0] * 100)
        
        client_info_table = pd.DataFrame({
            'Caractéristiques' : ['Âge', 'Nombre d\'enfants','Revenus totaux', 'Ancienneté', 'Prix du bien', 'Taux de remboursement'],
            'Données' : [age_client, nb_children_client, income_client, tenure_client, goodprice_client, rate_client]
        })
        return client_info_table

# fonction pour récupérer les shap_values
def get_client_shap_values(current_client_id, data, model, explainer, column_names, columns_to_display):

    scaler = model.named_steps['scaler']
    X = get_client_data(current_client_id, data, column_names)
    X_transformed = scaler.transform(X)
    shap_values = explainer(X_transformed, check_additivity = False).values
    data_shap = pd.DataFrame(shap_values, columns = column_names)

    return data_shap[columns_to_display]

# fonction pour afficher le détail de la prédiction avec les shap values
def plot_shap_values(data_shap, current_client_id):
    # Déterminer les couleurs : rouge pour les positifs, vert pour les négatifs
    colors = data_shap['SHAP Value'].apply(lambda x: 'red' if x > 0 else 'green')

    # Ajouter un '+' devant les valeurs positives et formater les valeurs
    text_values = data_shap['SHAP Value'].apply(lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}")
    
    # Créer le graphique avec Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=[feature.replace('_', ' ') for feature in data_shap['Feature']],
        x=data_shap['SHAP Value'],
        marker_color=colors,
        text=text_values,  # Formater les valeurs à 2 décimales
        textposition='outside',  # Positionner le texte à l'extérieur des barres
        textfont = dict(size = 14, family = "Impact"),
        textfont_color = colors.tolist(),
        orientation='h'  # Orientation horizontale
    ))
    # Calculer la longueur maximale des labels pour ajuster les marges
    max_label_length = max(data_shap['Feature'].apply(lambda x: len(x)))
    margin_left = 20 + max_label_length * 2  # Ajuster le facteur multiplicateur pour plus ou moins d'espace

    # Mettre à jour la mise en page pour enlever les titres des axes X et Y
    fig.update_layout(
        title={
        'text': f"Détail de la prédiction pour le client {int(current_client_id)}",
        'x': 0.5,
        'xanchor': 'center',
        'font': dict(size = 24, family = 'Impact'),
        'pad': dict(t=50)
        },
        xaxis_title=None,  # Supprimer le titre de l'axe X
        yaxis_title=None,  # Supprimer le titre de l'axe Y
        xaxis = dict(
            tickfont = dict(size=14, family = 'Impact'),
            range = [-1.2, 1.2]
        ),
        yaxis = dict(
            tickfont = dict(size=16, family = 'Impact'),
        ),
        margin=dict(l=margin_left, r=20, t=100, b=10),
        title_pad = dict(t = 80),
        title_xanchor = 'center',
        title_yanchor = 'top'
    )
    # Afficher le plot
    return fig

# fonction pour le scatterplot
def scatter_plot(data, x_axis, y_axis, current_client_id):
    
    # Séparer les clients sélectionnés et les autres
    selected_clients = data[data['SK_ID_CURR'] == current_client_id]
    other_clients = data[data['SK_ID_CURR'] != current_client_id]

    fig = go.Figure()
    
    # Tracé pour les autres clients
    fig.add_trace(go.Scatter(
        x=other_clients[x_axis],
        y=other_clients[y_axis],
        mode='markers',
        marker=dict(
            size=8,
            color='#add8e6'  # Couleur des autres clients
        ),
        name='Autres clients'  # Nom dans la légende
    ))
    
    # Tracé pour le client sélectionné
    fig.add_trace(go.Scatter(
        x=selected_clients[x_axis],
        y=selected_clients[y_axis],
        mode='markers',
        marker=dict(
            size=15,
            color='#f93c13'  # Couleur du client sélectionné
        ),
        name=f"Client {int(current_client_id)}"  # Nom dans la légende
    ))
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title={
            'text': f"{x_axis.replace('_',' ')} vs {y_axis.replace('_',' ')}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'family': 'Impact'}
        },
        xaxis_title=x_axis.replace('_', ' '),
        yaxis_title=y_axis.replace('_', ' '),
        xaxis_title_font=dict(size=18, family="Impact"),
        yaxis_title_font=dict(size=18, family="Impact"),
        xaxis = dict(
            tickfont=dict(size=14, family='Impact')
        ),
        yaxis = dict(
            tickfont=dict(size=14, family='Impact')
        ),
        legend=dict(
            title_font=dict(size=16, family = 'Impact'),  # Taille et gras du titre de la légende
            title=dict(
                text="Type de points",
                font=dict(size=16, weight="bold"),
                side="top center"
            ),
            font = dict(size = 14, family = 'Impact'),
        ),
        title_pad = dict(t = 80),
        title_xanchor = 'center',
        title_yanchor = 'top'
    )
    
    return fig

# fonction pour envoyer la requête au modèle MLflow
def predict(input_data, model):   
    try:    
        predictions = model.predict_proba(input_data)
        return predictions
    except Exception as e:
        st.error(f"Erreur rencontrée lors de la prédiction : {e}")
        return None