import os
import time
import hmac
import hashlib
import requests
from decimal import Decimal
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

def build_grid(start_price, end_price, levels):
    step = (end_price - start_price) / (levels - 1)
    return [round(start_price + i * step, 4) for i in range(levels)]

def distribute_capital(capital, levels):
    amount_per_level = capital / levels
    return [round(amount_per_level, 6) for _ in range(levels)]

def main():
    print("🔁 Lancement du Grid Trading Bot...")

    # Saisie interactive
    symbol = input("🪙 Quel token veux-tu trader ? (ex: BTC-USDC) : ").strip().upper()
    total_capital = Decimal(input("💰 Capital à allouer (en USDC) : "))
    start_price = Decimal(input("📈 Prix minimum : "))
    end_price = Decimal(input("📉 Prix maximum : "))
    levels = int(input("📊 Nombre de grilles : "))

    # Construction de la grille
    grid = build_grid(start_price, end_price, levels)
    allocations = distribute_capital(total_capital, levels)

    print("\n📋 Stratégie Grid Trading :")
    for i in range(levels):
        print(f"Grille {i+1}: Prix {grid[i]} USDC → Allocation {allocations[i]} USDC")

    confirmation = input("\n✅ Confirmer le placement des ordres ? (o/n) : ").strip().lower()
    if confirmation != "o":
        print("❌ Annulé par l'utilisateur.")
        return

    # Placement des ordres d'achat sur chaque niveau de grille
    for i in range(levels):
        price = grid[i]
        # On suppose que l'allocation est en USDC, donc quantité = allocation/prix
        quantity = float(allocations[i]) / float(price)
        result = place_order(symbol, "buy", price, quantity)
        print("Réponse API :", result)
        time.sleep(1)  # Petite pause pour éviter le spam API

if __name__ == "__main__":
    main()
