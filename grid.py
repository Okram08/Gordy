import os
import requests
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
WALLET_ADDRESS = os.getenv("HL_WALLET_ADDRESS")

def get_spot_balance():
    """Récupère le solde USDC SPOT via l'API native Hyperliquid"""
    url = "https://api.hyperliquid.xyz/info"
    payload = {
        "type": "spotClearinghouseState",
        "user": WALLET_ADDRESS
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for asset in data.get("balances", []):
                if asset.get("coin") == "USDC":
                    # La valeur 'total' est une string, on la convertit en float
                    return float(asset.get("total", "0"))
            return 0.0
        else:
            print(f"Erreur HTTP {response.status_code}: {response.text}")
            return 0.0
    except Exception as e:
        print(f"Erreur API: {e}")
        return 0.0

def main():
    if not WALLET_ADDRESS or len(WALLET_ADDRESS) != 42 or not WALLET_ADDRESS.startswith("0x"):
        print("Adresse de wallet non renseignée ou invalide.")
        return
    solde_usdc = get_spot_balance()
    print(f"Votre solde SPOT USDC : {solde_usdc:.6f}")

if __name__ == "__main__":
    main()
