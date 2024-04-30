import streamlit as st
import pandas as pd
import joblib
import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Chargement du modèle
modele = joblib.load('meilleur_modele.joblib')

# Fonction pour effectuer les prédictions
def make_predictions(data):
    predictions = modele.predict(data)
    return predictions

# Fonction pour calculer les métriques
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

# Afficher le formulaire de prédiction
st.sidebar.markdown("# Formulaire de Prédiction")

# Widgets pour le formulaire de prédiction (partie gauche)
with st.sidebar:
    longitude = st.number_input("Longitude :", value=0.0)
    latitude = st.number_input("Latitude :", value=0.0)
    housing_median_age = st.number_input("Âge médian du logement :", value=0)
    total_rooms = st.number_input("Nombre total de pièces :", value=0)
    total_bedrooms = st.number_input("Nombre total de chambres à coucher :", value=0)
    population = st.number_input("Population :", value=0)
    households = st.number_input("Ménages :", value=0)
    median_income = st.number_input("Revenu médian :", value=0.0)
    ocean_proximity = st.selectbox("Proximité de l'océan :", ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])
    if st.button("Prédire"):
        user_data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'ocean_proximity': [ocean_proximity]
        })
        preprocessed_data = preprocessing.preprocess_data(user_data)
        prediction = make_predictions(preprocessed_data)
        st.write(f"Prédiction : {prediction}")

st.markdown("# Prédictions Immo 1990 CA")

# Afficher le titre pour l'importation de fichiers
st.markdown("## Importation de Fichiers")

# Widgets pour l'importation de fichiers (partie droite)
uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    original_data = data

    # Prétraiter les données
    preprocessed_data = preprocessing.preprocess_data(original_data)

    # Utiliser les colonnes sélectionnées
    selected_features = ['distance_to_sf', 'distance_to_la', 'distance_to_sd', 'housing_median_age', 'latitude', 'longitude', 'centre_ville', 'population', 'median_income', 'is_inland', 'total_bedrooms', 'total_rooms', 'households']
    X = preprocessed_data[selected_features]

    # Faire les prédictions
    predictions = make_predictions(X)

    # Calculer les métriques
    y_true = preprocessed_data['median_house_value']  # Utiliser la colonne 'median_house_value' des données prétraitées
    rmse, r2 = calculate_metrics(y_true, predictions)

    # Afficher les résultats
    st.write("RMSE:", rmse)
    st.write("R2:", r2)
