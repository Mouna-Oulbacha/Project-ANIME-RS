# import os
# from dotenv import load_dotenv
# from flask import Flask
# from extensions import mongo, bcrypt
# from routes.auth import auth_bp
# from routes.main import main_bp
# from routes.recommender_routes import recommender_bp
# from routes.main import main_bp

# # Charger les variables d'environnement
# load_dotenv()

# # Initialisation de l'application Flask
# app = Flask(__name__)

# app.secret_key = "mmmmm1234yyyttttlcmslxlancikoqqo"  # Assurez-vous qu'elle est unique et sécurisée


# # Configurations Flask
# app.config['MONGO_URI'] = "mongodb://localhost:27017/anime_recommendation_db"
# app.secret_key = os.getenv("SECRET_KEY", "clé_secrète_par_défaut")

# # Initialisation des extensions
# mongo.init_app(app)
# bcrypt.init_app(app)


# # Enregistrement des Blueprints
# # Enregistrement des Blueprints
# app.register_blueprint(auth_bp, url_prefix='/auth')  # URL pour l'authentification
# app.register_blueprint(main_bp, url_prefix='/')  # URL pour les routes principales
# app.register_blueprint(recommender_bp, url_prefix='/recommendations')


# if __name__ == '__main__':
#     app.run(debug=True)


import os
from flask import Flask, render_template
from extensions import mongo, bcrypt
from routes.auth import auth_bp
from routes.main import main_bp 
from routes.recommender_routes import recommender_bp

# Initialisation de l'application Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "votre_clé_secrète")
app.config['MONGO_URI'] = "mongodb://localhost:27017/anime_recommendation_db"

# Initialisation des extensions
mongo.init_app(app)
bcrypt.init_app(app)

# Enregistrement des Blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(recommender_bp, url_prefix='/recommendations')
app.register_blueprint(main_bp)


@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

