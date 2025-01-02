import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Huber
from tensorflow.keras.models import load_model

import pandas as pd

def calculate_average_ratings(ratings_csv_path):
    """
    Calculer les notes moyennes pour chaque anime à partir d'un fichier CSV.

    Args:
        ratings_csv_path (str): Chemin du fichier CSV contenant les ratings.

    Returns:
        dict: Un dictionnaire où les clés sont les anime_id et les valeurs sont les notes moyennes.
    """
    # Charger le fichier CSV
    ratings_df = pd.read_csv(ratings_csv_path)

    # Vérifier si les colonnes nécessaires existent
    if 'anime_id' not in ratings_df.columns or 'rating' not in ratings_df.columns:
        raise ValueError("Le fichier CSV doit contenir les colonnes 'anime_id' et 'rating'.")

    # Calculer les notes moyennes
    average_ratings = ratings_df.groupby('anime_id')['rating'].mean().to_dict()

    return average_ratings


def recommend_for_new_user(ratings_csv_path, top_n=10):
    """
    Recommander des animes enrichis pour les nouveaux utilisateurs.
    """
    try:
        # Charger les caractéristiques normalisées
        file_path = os.path.join(os.getcwd(), 'model_outputs', 'anime_feature_map.pkl')
        with open(file_path, 'rb') as f:
            anime_feature_map = pickle.load(f)

        # Charger les notes moyennes
        average_ratings = calculate_average_ratings(ratings_csv_path)

        # Trier les animes par score
        sorted_anime_ids = sorted(average_ratings, key=average_ratings.get, reverse=True)
        top_anime_ids = sorted_anime_ids[:top_n]

        # Charger les données brutes
        anime_raw = pd.read_csv('model_outputs/anime.csv')
        anime_raw_map = anime_raw.set_index('anime_id').to_dict('index')

        # Ajouter les détails enrichis
        recommendations = []
        for anime_id in top_anime_ids:
            raw_details = anime_raw_map.get(anime_id, {})
            
            # Convertir explicitement toutes les valeurs qui pourraient être NaN
            rating = average_ratings.get(anime_id)
            if pd.isna(rating):  # Vérifie si c'est NaN
                rating = "N/A"
            else:
                rating = float(rating)  # Convertit en float si ce n'est pas NaN

            members = raw_details.get("members", 0)
            if pd.isna(members):
                members = 0
            else:
                members = int(members)

            recommendations.append({
                "anime_id": int(anime_id),
                "name": str(raw_details.get("name", "Unknown")),
                "type": str(raw_details.get("type", "Unknown")),
                "genre": raw_details.get("genre", "").split(", ") if raw_details.get("genre") else [],
                "rating": rating,
                "members": members
            })

        return recommendations
    except Exception as e:
        print(f"Error in recommend_for_new_user: {str(e)}")
        return []

def recommend_for_user_from_csv(
    model, csv_file, user_id, anime_feature_map, anime_idx_map, df_anime, df_rating, top_n=10
):
    """
    Recommander des animes pour un utilisateur basé sur ses évaluations dans un fichier CSV.

    Arguments :
    - model : Le modèle de recommandation chargé depuis un fichier sauvegardé.
    - csv_file : Chemin du fichier CSV contenant les évaluations.
    - user_id : ID de l'utilisateur pour lequel on veut recommander.
    - anime_feature_map : Dictionnaire des fonctionnalités des animes.
    - anime_idx_map : Dictionnaire des indices des animes.
    - df_anime : DataFrame contenant les informations des animes.
    - df_rating : DataFrame contenant les évaluations pour calculer la popularité.
    - top_n : Nombre de recommandations à générer.

    Retourne :
    - Liste des recommandations enrichies.
    """
    # Charger les évaluations depuis le fichier CSV
    user_ratings = pd.read_csv(csv_file)

    # Filtrer pour récupérer uniquement les évaluations de cet utilisateur
    user_ratings = user_ratings[user_ratings['user_id'] == user_id]

    if user_ratings.empty:
        print(f"Aucune évaluation trouvée pour l'utilisateur {user_id}.")
        return []

    # Ajouter les fonctionnalités des animes évalués
    user_ratings['anime_features'] = user_ratings['anime_id'].apply(lambda x: anime_feature_map.get(x))
    user_ratings = user_ratings[~user_ratings['anime_features'].isna()]

    # Calculer un vecteur utilisateur moyen pondéré par les notes
    weighted_features = [
        np.array(features) * rating
        for features, rating in zip(user_ratings['anime_features'], user_ratings['rating'])
    ]
    #user_feature_vector = np.mean(weighted_features, axis=0)

    # Préparer les données pour les prédictions
    anime_ids = df_anime['anime_id'].tolist()
    anime_indices = [anime_idx_map.get(a_id) for a_id in anime_ids]
    valid_anime_ids = [a_id for a_id, idx in zip(anime_ids, anime_indices) if idx is not None]
    valid_anime_indices = [idx for idx in anime_indices if idx is not None]
    features = np.array([anime_feature_map[a_id] for a_id in valid_anime_ids])

    # Prédire les notes
    user_inputs = np.zeros(len(valid_anime_indices))  # ID utilisateur temporaire
    anime_inputs = np.array(valid_anime_indices)
    predictions = model.predict([user_inputs, anime_inputs, features], verbose=0)

    # Normaliser les prédictions
    predictions = normalize_predictions(predictions.flatten())

    # Calculer la diversité et la pertinence
    already_rated = set(user_ratings['anime_id'].tolist())
    recommended_animes = []
    for anime_id, pred in zip(valid_anime_ids, predictions):
        if anime_id in already_rated:
            continue

        # Score de diversité basé sur la popularité
        diversity_score = get_anime_diversity_score(anime_id, df_rating)

        # Similarité des genres (utiliser le genre moyen de l'utilisateur)
        user_genres = df_anime[df_anime['anime_id'].isin(already_rated)]['genre']
        user_genres_combined = ','.join(map(str, user_genres.dropna().tolist()))
        anime_genres = str(df_anime[df_anime['anime_id'] == anime_id]['genre'].values[0])
        genre_similarity = get_genre_similarity(user_genres_combined, anime_genres)

        # Score final
        final_score = pred * genre_similarity * diversity_score
        recommended_animes.append((anime_id, final_score))

    # Trier les recommandations
    recommended_animes.sort(key=lambda x: x[1], reverse=True)

    # Enrichir les résultats avec des détails depuis df_anime
    enriched_recommendations = []
    print(f"Recommandations pour l'utilisateur {user_id} :")
    for anime_id, score in recommended_animes[:top_n]:
        anime_info = df_anime[df_anime['anime_id'] == anime_id].iloc[0]
        anime_name = anime_info['name']
        anime_genre = anime_info['genre']
        anime_rating = anime_info['rating'] if 'rating' in anime_info else 'N/A'
        anime_type = anime_info['type'] if 'type' in anime_info else 'Unknown'
  

        print(f"{anime_name} | Type: {anime_type} | Genre: {anime_genre} | Note: {anime_rating:.2f} | Score: {score:.2f}")
        enriched_recommendations.append({
            'anime_id': anime_id,
            'name': anime_name,
            'type': anime_type,
            'genre': anime_genre,
            'rating': anime_rating,
            'score': score,
            
        })

    return enriched_recommendations
# Fonctions auxiliaires
def normalize_predictions(predictions, min_val=0.1, max_val=0.9):
    normalized = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    return normalized * (max_val - min_val) + min_val

def get_anime_diversity_score(anime_id, df_rating):
    popularity = len(df_rating[df_rating['anime_id'] == anime_id])
    return 1 / (1 + np.log1p(popularity))

def get_genre_similarity(anime1_genres, anime2_genres):
    if isinstance(anime1_genres, str) and isinstance(anime2_genres, str):
        genres1 = set(g.strip() for g in anime1_genres.split(','))
        genres2 = set(g.strip() for g in anime2_genres.split(','))
        return len(genres1.intersection(genres2)) / len(genres1.union(genres2))
    return 0