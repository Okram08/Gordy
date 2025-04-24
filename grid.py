import os
import time
import hmac
import hashlib
import requests
from decimal import Decimal
from dotenv import load_dotenv

# Paramètres de la grille
GRID_LEVELS = 3
GRID_SPREAD = Decimal("0.01")  # 1% au-dessus et en dessous du prix spot

# Charger les variables d'environnement
load_dotenv()
API_KEY = os.getenv("HL_API_KEY")
API_SECRET = os.getenv("HL_API_SECRET")
BASE_URL = "https://api.hyperliquid.xyz"

if not API_KEY or not API_SECRET:
    print("❌ Clé API ou Secret API introuvable. Vérifie ton fichier .env !")
    exit(1)

def sign_request(secret, payload):
    return hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def get_spot_price(symbol):
    # Endpoint Hyperliquid pour le prix spot (à adapter selon la doc officielle)
    url = BASE_URL + "/info"
    payload = {
        "type": "spotPrice",
        "coin": symbol.split('-')[0]
    }
    response = requests.post(url, json=payload)
    data = response.json()
    # On suppose que le prix spot est dans data['price']
    price = Decimal(str(data.get('price')))
    return price

def place_order(symbol, side, price, quantity):
    endpoint = "/order"  # À adapter selon la doc Hyperliquid
    url = BASE_URL + endpoint

    timestamp = str(int(time.time() * 1000))
    body = {
        "symbol": symbol,
        "side": side,
        "price": str(price),
        "quantity": str(quantity),
        "timestamp": timestamp
    }
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

def build_grid(center_price, levels, spread):
    # Construit une grille centrée sur le prix spot, écartée de +/- spread*100 %
    prices = []
    for i in range(levels):
        offset = (i - (levels // 2)) * spread
        grid_price = center_price * (1 + offset)
        prices.append(grid_price.quantize(Decimal("0.0001")))
    return prices

def distribute_capital(capital, levels):
    amount_per_level = capital / levels
    return [round(amount_per_level, 6) for _ in range(levels)]

def main():
    print("🔁 Lancement du Grid Trading Bot...")

    # Saisie interactive minimale
    symbol = input("🪙 Quelle paire veux-tu trader ? (ex: BTC-USDC) : ").strip().upper()
    total_capital = Decimal(input("💰 Capital à allouer (en USDC) : "))

    # Récupère le prix spot automatiquement
    spot_price = get_spot_price(symbol)
    print(f"📈 Prix spot actuel pour {symbol} : {spot_price} USDC")

    # Construit la grille automatiquement autour du prix spot
    grid = build_grid(spot_price, GRID_LEVELS, GRID_SPREAD)
    allocations = distribute_capital(total_capital, GRID_LEVELS)

    print("\n📋 Stratégie Grid Trading :")
    for i in range(GRID_LEVELS):
        print(f"Grille {i+1}: Prix {grid[i]} USDC → Allocation {allocations[i]} USDC")

    confirmation = input("\n✅ Confirmer le placement des ordres ? (o/n) : ").strip().lower()
    if confirmation != "o":
        print("❌ Annulé par l'utilisateur.")
        return

    # Placement des ordres d'achat sur chaque niveau de grille
    for i in range(GRID_LEVELS):
        price = grid[i]
        quantity = float(allocations[i]) / float(price)
        result = place_order(symbol, "buy", price, quantity)
        print("Réponse API :", result)
        time.sleep(1)

if __name__ == "__main__":
    main()
