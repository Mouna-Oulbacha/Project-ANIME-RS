class Anime:
    def __init__(self, mongo):
        self.mongo = mongo

    def get_top_animes(self, limit=10):
        """ Récupère les animes avec les meilleures évaluations moyennes """
        return list(self.mongo.db.animes.find().sort("rating", -1).limit(limit))

    def find_anime_by_id(self, anime_id):
        """ Recherche un anime spécifique """
        return self.mongo.db.animes.find_one({"anime_id": anime_id})
