import jwt
from flask import request, jsonify

def decode_jwt(token, secret_key):
    """
    Décoder un token JWT et valider son contenu.
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return {"error": "Token expiré."}
    except jwt.InvalidTokenError:
        return {"error": "Token invalide."}
