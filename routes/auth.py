# from flask import Blueprint, render_template, request, redirect, url_for, session, flash
# import app
# from extensions import mongo, bcrypt  # Importer MongoDB et Bcrypt initialisés
# from models.User import User
# import jwt
# import datetime
# from flask import jsonify


# auth_bp = Blueprint('auth', __name__)  # Donnez un nom unique

# user_service = User(mongo, bcrypt)  # Instanciation de la classe User

# @auth_bp.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']
#         confirm_password = request.form['confirm_password']

#         # Validation des mots de passe
#         if password != confirm_password:
#             return "Les mots de passe ne correspondent pas!"

#         # Vérifier si l'utilisateur existe déjà
#         if user_service.find_by_username(username):
#             return "Utilisateur déjà existant!"
#         if user_service.find_by_email(email):
#             return "E-mail déjà utilisé!"

#         # Créer un nouvel utilisateur
#         user_service.create_user(username, email, password)
#         return redirect(url_for('auth.login'))

#     return render_template('register.html')


# # @auth_bp.route('/login', methods=['GET', 'POST'])
# # def login():
# #     if request.method == 'POST':
# #         username = request.form['username']
# #         password = request.form['password']

# #         # Recherche de l'utilisateur par nom d'utilisateur
# #         user = user_service.find_by_username(username)
# #         if not user or not user_service.check_password(user['password'], password):
# #             flash("Nom d'utilisateur ou mot de passe incorrect!", "error")
# #             return render_template('login.html')

# #         # Création de la session utilisateur
# #         session['username'] = user['username']
# #         flash(f"Bienvenue, {user['username']}!", "success")
# #         return redirect(url_for('main.dashboard'))

# #     return render_template('login.html')


# @auth_bp.route('/login', methods=['POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']

#         # Recherche de l'utilisateur par nom d'utilisateur
#         user = user_service.find_by_username(username)
#         if not user or not user_service.check_password(user['password'], password):
#             return jsonify({"error": "Nom d'utilisateur ou mot de passe incorrect!"}), 403

#         # Générer un token JWT
#         token = jwt.encode(
#             {
#                 "username": username,
#                 "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # Expire dans 1 heure
#             },
#             app.secret_key,
#             algorithm="HS256"
#         )

#         # Renvoyer le token au client
#         return jsonify({"token": token})



# @auth_bp.route('/logout')
# def logout():
#     session.clear()
#     flash("Vous avez été déconnecté.", "success")
#     return redirect(url_for('auth.login'))



import jwt
import datetime
from flask import Blueprint, app, current_app, redirect, render_template, request, jsonify, url_for
from extensions import mongo, bcrypt
from models.User import User

auth_bp = Blueprint('auth', __name__)  # Initialisation du Blueprint
user_service = User(mongo, bcrypt)


@auth_bp.route('/logout', methods=['GET'])
def logout():
    # Supprimer le token de l'utilisateur, par exemple en vidant le cookie
    response = redirect(url_for('auth.login'))
    response.delete_cookie('token')  # Supprimez le cookie contenant le token JWT
    return response


# @auth_bp.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')

#         # Vérifier l'utilisateur dans la base de données
#         user = user_service.find_by_username(username)
#         if not user or not user_service.check_password(user['password'], password):
#            return jsonify({"error": "Nom d'utilisateur ou mot de passe incorrect"}), 403
        
        
        
#         # Générer un token JWT
#         token = jwt.encode(
#             {
#                 "user_id": user['user_id'],  # Ajouter le user_id ici
#                 "username": username,
#                 "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
#             },
#             current_app.config['SECRET_KEY'],  # Utilisation de current_app pour accéder à la configuration
#             algorithm="HS256"
#         )
#         # Imprimer le token dans le terminal Flask
#         print("Token généré :", token)
        
#         # Sauvegarder le token dans un cookie
#         response = redirect(url_for('home'))  # Redirection vers la route 'home'
#         response.set_cookie('token', token, httponly=True, secure=False)  # Enregistrer le token JWT dans un cookie
#         return response 

#     return render_template('login.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Vérifier l'utilisateur dans la base de données
        user = user_service.find_by_username(username)
        if not user or not user_service.check_password(user['password'], password):
            return jsonify({"error": "Nom d'utilisateur ou mot de passe incorrect"}), 403
            
        # Générer un token JWT
        token = jwt.encode(
            {
                "user_id": user['user_id'],
                "username": username,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
            },
            current_app.config['SECRET_KEY'],
            algorithm="HS256"
        )
        print("Token généré :", token)
        
        return jsonify({
            "success": True,
            "token": token
        })
    
    return render_template('login.html')
    


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validation des mots de passe
        if password != confirm_password:
            return jsonify({"error": "Les mots de passe ne correspondent pas."}), 400

        # Vérification si l'utilisateur existe déjà
        if user_service.find_by_username(username):
            return jsonify({"error": "Le nom d'utilisateur est déjà utilisé."}), 400

        if user_service.find_by_email(email):
            return jsonify({"error": "L'adresse e-mail est déjà utilisée."}), 400

        # Créer l'utilisateur
        user_service.create_user(username, email, password)
        return redirect(url_for('auth.login'))
    

        

    return render_template('register.html')  # Si c'est un GET, afficher le formulaire d'inscription
