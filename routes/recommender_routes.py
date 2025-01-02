import os
import pickle
from flask import Blueprint, jsonify, request, current_app
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.auth import decode_jwt
from models.recommendation_model import recommend_by_genre, recommend_for_new_user, recommend_for_user_from_csv
from models.User import User
from extensions import mongo, bcrypt
import os
import pickle
from flask import Blueprint, jsonify, request, current_app
from pymongo import MongoClient

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

recommender_bp = Blueprint('recommender', __name__)
user_service = User(mongo, bcrypt)


@recommender_bp.route('/recommend', methods=['GET'])
def recommend():
    """
    Route pour recommander des animes enrichis pour tout utilisateur.
    """
    try:
        recommendations = recommend_for_new_user(ratings_csv_path='model_outputs/rating.csv', top_n=7)
        # Nettoyer une dernière fois les données pour le JSON
        cleaned_recommendations = []
        for rec in recommendations:
            cleaned_rec = {
                "anime_id": int(rec["anime_id"]),
                "name": str(rec["name"]),
                "type": str(rec["type"]),
                "genre": list(rec["genre"]),
                "rating": "N/A" if rec["rating"] == "N/A" else float(rec["rating"]),
                "members": int(rec["members"])
            }
            cleaned_recommendations.append(cleaned_rec)
        
        return jsonify({"recommendations": cleaned_recommendations})
    except Exception as e:
        print(f"Error in recommend route: {str(e)}")
        return jsonify({"error": str(e), "recommendations": []}), 500
    

# @recommender_bp.route('/rate', methods=['POST'])
# def rate_anime():
#     """
#     Permet à un utilisateur de noter un anime et régénère ses recommandations.
#     """
#     # Vérification du token JWT
#     auth_header = request.headers.get('Authorization')
#     if not auth_header or not auth_header.startswith("Bearer "):
#         return jsonify({"error": "Token manquant ou invalide."}), 403

#     token = auth_header.split(" ")[1]
#     payload = decode_jwt(token, current_app.config['SECRET_KEY'])

#     if "error" in payload:
#         return jsonify(payload), 403

#     username = payload["username"]

#     # Extraction des données
#     data = request.json
#     anime_id = data.get('anime_id')
#     rating = data.get('rating')

#     if not anime_id or not rating:
#         return jsonify({"error": "L'anime ID et la note sont requis."}), 400

#     if not (1 <= rating <= 10):
#         return jsonify({"error": "La note doit être comprise entre 1 et 10."}), 400

#     # Récupération du user_id
#     user_id = user_service.get_user_id_by_username(username)
#     if not user_id:
#         return jsonify({"error": "Utilisateur introuvable."}), 404

#     # Ajout de l'évaluation dans le fichier CSV
#     user_service.add_rating(user_id, anime_id, rating)

#     return jsonify({
#         "message": "Note ajoutée avec succès.",
#     })






# Connexion MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["anime_recommendation"]
users_collection = db["users"]

@recommender_bp.route('/rate', methods=['POST'])
def rate_anime():
    """
    Permet à un utilisateur de noter un anime et régénère ses recommandations.
    """
    # Vérification du token JWT
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Token manquant ou invalide."}), 403

    token = auth_header.split(" ")[1]
    payload = decode_jwt(token, current_app.config['SECRET_KEY'])

    if "error" in payload:
        return jsonify(payload), 403
    
    username = payload["username"]

    # Extraire le user_id depuis le token
    user_id = payload.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID manquant dans le token."}), 403
    print(user_id)



    # Extraction des données
    data = request.json
    anime_id = data.get('anime_id')
    rating = data.get('rating')

    if not anime_id or not rating:
        return jsonify({"error": "L'anime ID et la note sont requis."}), 400

    if not (1 <= rating <= 10):
        return jsonify({"error": "La note doit être comprise entre 1 et 10."}), 400
    
    # Ajouter la note
    user_service.add_anime_rating(username, anime_id, rating)

    # Ajout de l'évaluation dans le fichier CSV
    try:
        user_service.add_rating(user_id, anime_id, rating)
    except Exception as e:
        return jsonify({"error": "Erreur lors de l'ajout de la note.", "details": str(e)}), 



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
    
    # Génération de nouvelles recommandations
    try:
        recommendations = recommend_for_user_from_csv(
            model=model,
            csv_file="C:/Users/msi/Desktop/RS_Animes/data/df_user_anime.csv",
            user_id=user_id,
            anime_feature_map=anime_feature_map,
            anime_idx_map=anime_idx_map,
            df_anime=df_anime,
            df_rating=df_rating,
            top_n=10
        )

        # Mise à jour des recommandations dans MongoDB
        user_service.update_recommendations(user_id, recommendations)


        return jsonify({
            "message": "Note ajoutée avec succès.",
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": "Erreur lors de la génération des recommandations.", "details": str(e)}), 500
    

from flask import jsonify, request, current_app
from functools import wraps
import jwt

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({"error": "Authorization header is missing"}), 401
            
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Invalid authorization format. Must start with 'Bearer'"}), 401
            
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            request.user = payload
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
            
    return decorated

@recommender_bp.route('/getRecommandations', methods=['GET'])
@require_auth
def get_user_recommendations():
    """
    Returns recommendations for the current user.
    """
    try:
        # Get user_id from the decoded token (now available in request.user)
        user_id = request.user.get("user_id")
        if not user_id:
            return jsonify({"error": "User ID not found in token"}), 400

        # Get user recommendations from database
        user = mongo.db.users.find_one({'user_id': user_id})
        if not user:
            return jsonify({"error": "User not found"}), 404

        recommendations = user.get('recommendations', [])
        return jsonify({
            "success": True,
            "recommendations": recommendations
        })

    except Exception as e:
        # Log the error server-side
        current_app.logger.error(f"Error fetching recommendations: {str(e)}")
        return jsonify({
            "error": "An error occurred while fetching recommendations",
            "details": str(e)
        }), 500


@recommender_bp.route('/recommendByGenre', methods=['GET'])
def recommendByGenre():
    genre = request.args.get('genre', default="", type=str)
    df_anime = pd.read_csv("C:/Users/msi/Desktop/RS_Animes/model_outputs/anime.csv")
    
    # Get filtered recommendations
    filtered_recommendations = recommend_by_genre(df_anime, genre_filter=genre)
    
    return jsonify({"recommendations": filtered_recommendations})

