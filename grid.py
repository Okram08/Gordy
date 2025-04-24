import os
import json
import requests
from dotenv import load_dotenv
from decimal import Decimal

# Charger le .env
load_dotenv()

WALLET_ADDRESS = os.getenv("HL_WALLET_ADDRESS")
PRIVATE_KEY = os.getenv("HL_PRIVATE_KEY")  # Pas encore utilisé ici

# API URL
HL_API_URL = "https://api.hyperliquid.xyz/info"

def get_spot_balance(address):
    payload = {
        "type": "spotClearinghouseState",
        "user": address
    }

    try:
        response = requests.post(HL_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("balances", [])
    except Exception as e:
        print(f"❌ Erreur lors de la requête : {e}")
        return []

def build_grid(start_price, end_price, levels):
    step = (end_price - start_price) / (levels - 1)
    return [round(start_price + i * step, 4) for i in range(levels)]

def distribute_capital(capital, levels, allocation_type="equal"):
    if allocation_type == "equal":
        amount_per_level = capital / levels
        return [round(amount_per_level, 2)] * levels
    # Tu peux implémenter des répartitions personnalisées ici
    return []

def main():
    print("🔁 Lancement du Grid Trading Bot...")

    # Lire le solde
    balances = get_spot_balance(WALLET_ADDRESS)
    usdc_balance = next((Decimal(b["total"]) for b in balances if b["coin"] == "USDC"), Decimal("0"))
    print(f"💵 Solde USDC disponible : {usdc_balance} USDC")

    # Définir les paramètres de la grille
    total_capital = Decimal(input("💰 Capital à allouer (USDC) : "))
    start_price = Decimal(input("📈 Prix minimum : "))
    end_price = Decimal(input("📉 Prix maximum : "))
    levels = int(input("📊 Nombre de grilles : "))

    if total_capital > usdc_balance:
        print("❗ Capital supérieur à ton solde. Abandon.")
        return

    grid = build_grid(start_price, end_price, levels)
    allocations = distribute_capital(total_capital, levels)

    print("\n📋 Stratégie Grid Trading :")
    for i in range(levels):
        print(f"Grille {i+1}: Prix {grid[i]} USDC → Allocation {allocations[i]} USDC")

    # Prochaine étape : placer les ordres spot ici via l’API (à discuter 😉)

if __name__ == "__main__":
    main()
