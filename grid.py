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
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            for asset in data.get("assetPositions", []):
                if asset["coin"] == "USDC":
                    return float(asset["position"])
        return 0.0
    except Exception as e:
        print(f"Erreur API: {e}")
        return 0.0

def main():
    solde_usdc = get_spot_balance()
    print(f"Votre solde SPOT USDC : {solde_usdc:.2f}")

if __name__ == "__main__":
    main()
