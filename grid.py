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

def get_spot_price_coingecko(symbol):
    mapping = {
        "BTC-USDC": ("bitcoin", "usd"),
        "ETH-USDC": ("ethereum", "usd"),
        # Ajoute d'autres mappings si besoin
    }
    ids, vs = mapping.get(symbol, (None, None))
    if not ids or not vs:
        raise ValueError("Paire non supportée pour CoinGecko.")
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies={vs}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print("❌ Erreur CoinGecko :", resp.status_code, resp.text)
        exit(1)
    price = resp.json()[ids][vs]
    return Decimal(str(price))

def place_order(symbol, side, price, quantity):
    endpoint = "/order"  # À adapter selon la doc Hyperliquid si besoin
    url = BASE_URL + endpoint

    timestamp = str(int(time.time() * 1000))
    body = {
        "symbol": symbol,
        "side": side,
        "price": str(price),
        "quantity": str(quantity),
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
    print(f"🚀 Envoi de l'ordre LIVE: {body}")
    response = requests.post(url, json=body, headers=headers)
    try:
        response.raise_for_status()
        print("✅ Ordre envoyé avec succès.")
    except Exception as e:
        print("❌ Erreur lors de l'envoi de l'ordre:", e)
        print("Réponse brute :", response.text)
    return response.json()

def build_grid(center_price, levels, spread):
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

    symbol = input("🪙 Quelle paire veux-tu trader ? (ex: BTC-USDC) : ").strip().upper()
    total_capital = Decimal(input("💰 Capital à allouer (en USDC) : "))

    try:
        spot_price = get_spot_price_coingecko(symbol)
    except Exception as e:
        print(f"❌ Impossible de récupérer le prix spot pour {symbol} : {e}")
        return

    print(f"📈 Prix spot actuel pour {symbol} : {spot_price} USDC")

    grid = build_grid(spot_price, GRID_LEVELS, GRID_SPREAD)
    allocations = distribute_capital(total_capital, GRID_LEVELS)

    print("\n📋 Stratégie Grid Trading :")
    for i in range(GRID_LEVELS):
        print(f"Grille {i+1}: Prix {grid[i]} USDC → Allocation {allocations[i]} USDC")

    confirmation = input("\n✅ Confirmer le placement des ordres (LIVE) ? (o/n) : ").strip().lower()
    if confirmation != "o":
        print("❌ Annulé par l'utilisateur.")
        return

    for i in range(GRID_LEVELS):
        price = grid[i]
        quantity = float(allocations[i]) / float(price)
        result = place_order(symbol, "buy", price, quantity)
        print("Réponse API :", result)
        time.sleep(1)

if __name__ == "__main__":
    main()
