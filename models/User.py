import csv
import os
import pickle


class User:
    def __init__(self, mongo, bcrypt):
        self.mongo = mongo
        self.bcrypt = bcrypt
        self.ratings_csv_path = os.path.join(os.getcwd(), 'data', 'df_user_anime.csv')


        # Vérifier si le fichier CSV existe, sinon le créer avec un en-tête
        if not os.path.exists(self.ratings_csv_path):
            with open(self.ratings_csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['user_id', 'anime_id', 'rating'])  # En-têtes
    
    def get_next_user_id(self):
        """ Récupère le prochain user_id basé sur le fichier CSV """
        try:
            with open(self.ratings_csv_path, mode='r') as file:
              reader = csv.reader(file)
              rows = list(reader)
              if len(rows) > 1:
                 # Récupérer le dernier user_id
                last_user_id = int(rows[-1][0])
                return last_user_id + 1
              else:
                return 1
        except Exception:
            return 1


    def create_user(self, username, email, password):
        """ Crée un utilisateur avec un user_id incrémental """
        hashed_password = self.bcrypt.generate_password_hash(password).decode('utf-8')
        user_id = self.get_next_user_id()
        self.mongo.db.users.insert_one({
           'user_id': user_id,
           'username': username,
           'email': email,
           'password': hashed_password,
           'anime_ratings': [],  # Liste des animes notés
           'recommendations': []
        })
        return user_id
    
    def add_rating(self, user_id, anime_id, rating):
        """ Ajoute une évaluation au fichier CSV """
        with open(self.ratings_csv_path, mode='a', newline='') as file:
           writer = csv.writer(file)
           writer.writerow([user_id, anime_id, rating])

    def get_user_id_by_username(self, username):
        """
        Récupère le user_id à partir du username
        """
        user = self.mongo.db.users.find_one({'username': username})
        return user['user_id'] if user else None


    def find_by_username(self, username):
        """ Recherche par nom d'utilisateur """
        return self.mongo.db.users.find_one({'username': username})

    def find_by_email(self, email):
        """ Recherche par email """
        return self.mongo.db.users.find_one({'email': email})

    def check_password(self, stored_password, provided_password):
        """ Vérifie le mot de passe """
        return self.bcrypt.check_password_hash(stored_password, provided_password)
    
    def update_recommendations(self, user_id, recommendations):
        """Met à jour les recommandations d'un utilisateur dans MongoDB"""
        try:
            self.mongo.db.users.update_one(
                {'user_id': user_id},
                {'$set': {'recommendations': recommendations}}
            )
        except Exception as e:
            raise Exception(f"Erreur lors de la mise à jour des recommandations : {str(e)}")

    def get_recommendations(self, user_id):
        """Récupère les recommandations pour un utilisateur"""
        user = self.mongo.db.users.find_one({'user_id': user_id}, {'recommendations': 1})
        return user.get('recommendations', []) if user else []
    
    def add_anime_rating(self, username, anime_id, rating):
        """
        Ajouter une note pour un anime donné.
        """
        user = self.find_by_username(username)
        if not user:
            return {"error": "Utilisateur introuvable."}

        # Vérifier si l'anime a déjà une note, et la mettre à jour si nécessaire
        existing_rating = next((r for r in user['anime_ratings'] if r['anime_id'] == anime_id), None)
        if existing_rating:
            existing_rating['rating'] = rating
        else:
            user['anime_ratings'].append({'anime_id': anime_id, 'rating': rating})

        # Mettre à jour l'utilisateur dans la base de données
        self.mongo.db.users.update_one(
            {'username': username},
            {'$set': {'anime_ratings': user['anime_ratings']}}
        )
        return {"message": "Note ajoutée avec succès."}

    def get_user_ratings(self, username):
        """
        Récupérer les évaluations d'un utilisateur avec leur user_id inclus.
        """
        user = self.find_by_username(username)
        if not user:
           return {"error": "Utilisateur introuvable."}

        user_id = user['_id']  # Récupérer l'ID utilisateur
        ratings = user.get('anime_ratings', [])

        # Ajouter user_id à chaque évaluation
        for rating in ratings:
            rating['user_id'] = str(user_id)
        return ratings
 




