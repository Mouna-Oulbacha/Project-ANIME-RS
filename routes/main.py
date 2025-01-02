from flask import Blueprint, render_template, session, redirect, url_for

from config import MONGO_URI

from extensions import mongo, bcrypt


# Définir le blueprint
main_bp = Blueprint('main', __name__)  # Nom unique et clair



@main_bp.route('/')
def home():
    return render_template('index.html')



# Route principale après connexion
@main_bp.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    return render_template('dashboard.html', username=session['username'])

from flask import Blueprint, jsonify, session, request
from models.User import User
from models.recommendation_model import recommend_for_new_user


@main_bp.route('/recommend', methods=['GET'])
def recommend():
    """
    Route pour recommander des animes.
    """
    if 'username' not in session:
        return jsonify({"error": "Veuillez vous connecter pour obtenir des recommandations."}), 403

    username = session['username']
    user = User(mongo, bcrypt)
    user_ratings = user.get_user_ratings(username)

    if user_ratings:
        # Recommandations personnalisées
        recommendations = recommend_for_new_user(user_ratings)
    else:
        # Recommandations globales
        recommendations = recommend_for_new_user()

    return jsonify({"recommendations": recommendations})


@main_bp.route('/rate', methods=['POST'])
def rate_anime():
    """
    Permet à un utilisateur de noter un anime.
    """
    if 'username' not in session:
        return jsonify({"error": "Veuillez vous connecter pour noter un anime."}), 403

    username = session['username']
    data = request.json
    anime_id = data.get('anime_id')
    rating = data.get('rating')

    if not anime_id or not rating:
        return jsonify({"error": "Anime ID et note requis."}), 400

    user = User(MONGO_URI, bcrypt)
    user.add_anime_rating(username, anime_id, rating)

    return jsonify({"message": "Évaluation enregistrée avec succès."})
