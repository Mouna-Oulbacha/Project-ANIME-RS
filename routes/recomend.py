import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model


import tensorflow as tf

#

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
            'score': score
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

# def recommend_by_genre(df_anime, genre_filter, top_n=10):
#     # Filtrer les animes contenant le genre souhaité
#     filtered_anime = df_anime[df_anime['genre'].apply(lambda genres: genre_filter in genres)]
    
#     # Trier par rating et popularité
#     recommendations = filtered_anime.sort_values(['rating', 'members'], ascending=False).head(top_n)
    
#     print(f"Recommandations pour un nouvel utilisateur basé sur le genre '{genre_filter}' :")
#     for idx, row in recommendations.iterrows():
#         print(f"{row['name']} - Rating: {row['rating']:.2f}, Members: {row['members']:.2f}")
        

def recommend_by_genre(df_anime, genre_filter, top_n=10):
    """
    Recommends top animes based on a specified genre.

    Parameters:
        df_anime (pd.DataFrame): DataFrame containing anime information.
        genre_filter (str): The genre to filter animes by.
        top_n (int): The number of recommendations to return.

    Returns:
        None
    """
    # Handle missing or invalid data in the 'genre' column
    df_anime['type'] = df_anime['type'].fillna("").astype(str)

    # Filter animes containing the specified genre
    filtered_anime = df_anime[df_anime['type'].apply(
        lambda genres: genre_filter.lower() in genres.lower()  # Case-insensitive match
    )]

    # Sort by rating and popularity (members), descending
    recommendations = filtered_anime.sort_values(
        ['rating', 'members'], ascending=False
    ).head(top_n)

    # Print recommendations
    print(f"Recommendations for a new user based on the genre '{genre_filter}':")
    for idx, row in recommendations.iterrows():
        print(f"{row['name']} - Rating: {row['rating']:.2f}, Members: {row['members']}")


def normalize_predictions(predictions, min_val=0.1, max_val=0.9):
    """Normalise les scores prédits pour éviter les valeurs extrêmes"""
    normalized = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    return normalized * (max_val - min_val) + min_val

def get_anime_diversity_score(anime_id, df_rating):
    """Calcule un score de diversité basé sur la rareté de l'anime"""
    popularity = len(df_rating[df_rating['anime_id'] == anime_id])
    return 1 / (1 + np.log1p(popularity))

def get_genre_similarity(anime1_genres, anime2_genres):
    """Calcule la similarité entre les genres de deux animes"""
    if isinstance(anime1_genres, str) and isinstance(anime2_genres, str):
        genres1 = set(g.strip() for g in anime1_genres.split(','))
        genres2 = set(g.strip() for g in anime2_genres.split(','))
        return len(genres1.intersection(genres2)) / len(genres1.union(genres2))
    return 0

def get_improved_recommendations(user_id, model, df_anime, df_rating, user_idx_map, anime_idx_map, 
                               anime_feature_map, top_n=10, diversity_weight=0.2, novelty_weight=0.3):
    """
    Génère des recommandations améliorées avec diversité et nouveauté
    """
    # if user_id not in user_idx_map:
    #     return "Utilisateur non trouvé dans la base de données"
    
    user_idx = user_idx_map[user_id]
    
    # Obtenir les animes déjà notés par l'utilisateur
    rated_animes = set(df_rating[df_rating['user_id'] == user_id]['anime_id'])
    user_ratings = df_rating[df_rating['user_id'] == user_id]
    
    # Trouver les genres préférés de l'utilisateur
    user_favorite_genres = set()
    for anime_id in rated_animes:
        anime_genres = df_anime[df_anime['anime_id'] == anime_id]['genre'].iloc[0]
        if isinstance(anime_genres, str):
            user_favorite_genres.update(g.strip() for g in anime_genres.split(','))
    
    # Préparer les animes à prédire
    all_animes = set(anime_idx_map.keys())
    animes_to_predict = list(all_animes - rated_animes)
    
    # Préparer les données pour la prédiction
    user_idxs = np.array([user_idx] * len(animes_to_predict))
    anime_idxs = np.array([anime_idx_map[anime_id] for anime_id in animes_to_predict])
    anime_features = np.array([anime_feature_map[anime_id] for anime_id in animes_to_predict])
    
    # Faire les prédictions de base
    base_predictions = model.predict([user_idxs, anime_idxs, anime_features]).flatten()
    
    # Normaliser les prédictions
    predictions = normalize_predictions(base_predictions)
    
    # Calculer les scores finaux avec diversité et nouveauté
    final_scores = []
    seen_franchises = set()
    
    for anime_id, pred_score in zip(animes_to_predict, predictions):
        # Score de diversité
        diversity_score = get_anime_diversity_score(anime_id, df_rating)
        
        # Vérifier si l'anime fait partie d'une franchise déjà vue
        anime_name = df_anime[df_anime['anime_id'] == anime_id]['name'].iloc[0]
        franchise_name = anime_name.split(':')[0].split(' Season')[0]
        
        # Pénaliser les franchises répétées
        franchise_penalty = 0.7 if franchise_name in seen_franchises else 1.0
        seen_franchises.add(franchise_name)
        
        # Score final combiné
        final_score = (
            pred_score * (1 - diversity_weight - novelty_weight) +
            diversity_score * diversity_weight
        ) * franchise_penalty
        
        final_scores.append((anime_id, final_score))
    
    # Trier et filtrer les résultats
    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Préparer les recommandations
    recommendations = []
    seen_franchises = set()
    for anime_id, score in final_scores:
        anime_info = df_anime[df_anime['anime_id'] == anime_id].iloc[0]
        franchise_name = anime_info['name'].split(':')[0].split(' Season')[0]
        
        # Éviter trop de recommandations de la même franchise
        if len(recommendations) < top_n and franchise_name not in seen_franchises:
            recommendations.append({
                'anime_id': int(anime_id),
                'name': anime_info['name'],
                'predicted_score': float(score),
                'genre': anime_info['genre'],
                'type': anime_info['type']
            })
            seen_franchises.add(franchise_name)
    
    return recommendations

def print_improved_recommendations(recommendations):
    """Affiche les recommandations de manière plus lisible"""
    print("\nRecommandations d'anime améliorées :")
    print("-" * 80)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   Genre: {rec['genre']}")
        print(f"   Score prédit: {rec['predicted_score']:.2f}")
        print("-" * 80)

# Fonction de test
def test_improved_recommender(user_id, model, df_anime, df_rating, user_idx_map, anime_idx_map, anime_feature_map):
    """Teste le système de recommandation amélioré"""
    print(f"\nProfil de l'utilisateur {user_id}:")
    user_ratings = df_rating[df_rating['user_id'] == user_id].merge(
        df_anime[['anime_id', 'name', 'genre']], 
        on='anime_id'
    )
    print("\nAnimes déjà notés:")
    print(user_ratings[['name', 'rating', 'genre']].sort_values('rating', ascending=False).head())
    
    recommendations = get_improved_recommendations(
        user_id=user_id,
        model=model,
        df_anime=df_anime,
        df_rating=df_rating,
        user_idx_map=user_idx_map,
        anime_idx_map=anime_idx_map,
        anime_feature_map=anime_feature_map
    )
    
    print_improved_recommendations(recommendations)



if __name__ == "__main__":
    # Charger le modèle sauvegardé
    saved_model_path = "C:/Users/msi/Desktop/RS_Animes/data/best_model_weighted.h5"

    model = tf.keras.models.load_model(saved_model_path)



    # Charger les autres objets nécessaires
    with open('C:/Users/msi/Desktop/RS_Animes/data/anime_idx_map.pkl', 'rb') as f:
        anime_idx_map = pickle.load(f)

    

    with open('C:/Users/msi/Desktop/RS_Animes/data/user_idx_map.pkl', 'rb') as f:
        user_idx_map = pickle.load(f)

    with open('C:/Users/msi/Desktop/RS_Animes/data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('C:/Users/msi/Desktop/RS_Animes/data/anime_feature_map.pkl', 'rb') as f:
        anime_feature_map = pickle.load(f)

    df_anime = pd.read_csv("C:/Users/msi/Desktop/RS_Animes/model_outputs/anime.csv")
    df_rating = pd.read_csv("C:/Users/msi/Desktop/RS_Animes/model_outputs/rating.csv")
    df_user_anime = pd.read_csv("C:/Users/msi/Desktop/RS_Animes/data/df_user_anime.csv")
        # Générer des recommandations pour un utilisateur donné
    # recommendations = recommend_for_user_from_csv(
    #         model=model,
    #         csv_file="C:/Users/msi/Desktop/RS_Animes/data/df_user_anime.csv",
    #         user_id=1,
    #         anime_feature_map=anime_feature_map,
    #         anime_idx_map=anime_idx_map,
    #         df_anime=df_anime,
    #         df_rating=df_rating,
    #         top_n=10
    # )
    recommend_by_genre(df_anime, genre_filter="Movie")

