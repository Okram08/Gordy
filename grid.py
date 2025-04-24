import os
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

API_KEY = os.getenv("HL_API_KEY")
API_SECRET = os.getenv("HL_API_SECRET")
BASE_URL = "https://api.hyperliquid.xyz"

def sign_request(secret, payload):
    """Signature HMAC SHA256 du payload"""
    return hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def place_order(symbol, side, price, quantity):
    endpoint = "/order"  # À adapter selon la doc Hyperliquid
    url = BASE_URL + endpoint

    timestamp = str(int(time.time() * 1000))
    body = {
        "symbol": symbol,         # ex: "BTC-USDC"
        "side": side,             # "buy" ou "sell"
        "price": str(price),      # ex: "65000"
        "quantity": str(quantity),# ex: "0.01"
        "timestamp": timestamp
    }

    # Construction du payload pour la signature
    payload = "&".join([f"{k}={v}" for k, v in body.items()])
    signature = sign_request(API_SECRET, payload)

    headers = {
        "API-KEY": API_KEY,
        "SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    print(f"Envoi de l'ordre: {body}")
    response = requests.post(url, json=body, headers=headers)
    try:
        response.raise_for_status()
        print("✅ Ordre envoyé avec succès.")
    except Exception as e:
        print("❌ Erreur lors de l'envoi de l'ordre:", e)
    return response.json()

if __name__ == "__main__":
    # Exemple d'utilisation
    symbol = "BTC-USDC"
    side = "buy"
    price = 65000
    quantity = 0.01

    result = place_order(symbol, side, price, quantity)
    print("Réponse de l'API :", result)
