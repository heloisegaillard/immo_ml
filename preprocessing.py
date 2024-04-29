import pandas as pd
from geopy.distance import geodesic
from sklearn.preprocessing import  MinMaxScaler


def preprocess_data(df):
    # Supprimer la colonne 'Unnamed: 0' si elle existe
    try:
        df = df.drop('Unnamed: 0', axis=1)
    except:
        print('no unnamed')

    # Supprimer les lignes avec des valeurs manquantes dans 'total_bedrooms'
    df = df.dropna(subset=['total_bedrooms'])

    # Supprimer les lignes où 'households' est supérieur à 'population'
    df = df.drop(df[df['households'] > df['population']].index)

    # Ajouter une colonne 'is_inland' qui indique si le logement est à l'intérieur des terres
    df['is_inland'] = (df['ocean_proximity'] == 'INLAND').astype(int)

    # Supprimer la colonne 'ocean_proximity'
    df = df.drop(['ocean_proximity'], axis=1)

    # Calculer les distances vers les centres urbains
    def distance_to_location(row, location_center):
        point = (row['latitude'], row['longitude'])
        return geodesic(location_center, point).kilometers

    def calculate_distances(df):
        sf_center = (37.7749, -122.4194)
        la_center = (34.0522, -118.2437)
        sd_center = (32.7157, -117.1611)

        df['distance_to_sf'] = df.apply(distance_to_location, args=(sf_center,), axis=1)
        df['distance_to_la'] = df.apply(distance_to_location, args=(la_center,), axis=1)
        df['distance_to_sd'] = df.apply(distance_to_location, args=(sd_center,), axis=1)

    calculate_distances(df)
    # Définir les seuils de distance pour chaque ville
    seuil_sf = 100  # Seuil pour San Francisco (100 km)
    seuil_la = 200  # Seuil pour Los Angeles (200 km)
    seuil_sd = 300  # Seuil pour San Diego (300 km)

    # Créer une colonne centre_ville
    df['centre_ville'] = 0

    # Mettre à jour la colonne centre_ville
    df.loc[(df['distance_to_sf'] < seuil_sf) |
        (df['distance_to_la'] < seuil_la) |
        (df['distance_to_sd'] < seuil_sd), 'centre_ville'] = 1

    scaler = MinMaxScaler()
    numeric_features = ['distance_to_sf','distance_to_la','distance_to_sd','longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    return df

def preprocess_user_data(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity):
    # Créer un DataFrame avec les données saisies par l'utilisateur
    user_df = pd.DataFrame({
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

    # Prétraiter les données
    preprocessed_user_df = preprocess_data(user_df)

    return preprocessed_user_df
