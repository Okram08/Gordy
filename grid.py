import os
import json
import time
import requests
from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

# Charger les variables d'environnement
load_dotenv()
wallet_address = os.getenv("HL_WALLET_ADDRESS")
private_key = os.getenv("HL_PRIVATE_KEY")

# Configuration Hyperliquid
BASE_URL = "https://api.hyperliquid.xyz"
CHAIN_ID = 1  # Mainnet

def get_nonce():
    return int(time.time() * 1000)

def sign_message(message, private_key):
    account = Account.from_key(private_key)
    signature = account.sign_message(message)
    return signature.signature.hex()

def get_spot_balance():
    url = f"{BASE_URL}/info"
    payload = {
        "type": "spotUser",
        "user": wallet_address
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        for asset in data.get("assetPositions", []):
            if asset["coin"] == "USDC":
                return float(asset["position"])
    return 0.0

def place_spot_order(side, coin, sz, px):
    nonce = get_nonce()
    order = {
        "type": "order",
        "nonce": nonce,
        "coin": coin,
        "sz": str(sz),
        "price": str(px),
        "side": side,
        "isPostOnly": False
    }
    
    signature = sign_message(str(nonce), private_key)
    
    headers = {
        "Content-Type": "application/json",
        "X-Signature": signature,
        "X-Wallet-Address": wallet_address
    }
    
    response = requests.post(
        f"{BASE_URL}/exchange",
        headers=headers,
        json=order
    )
    
    return response.json()

def grid_trading_bot():
    symbol = "ETH"  # Paire SPOT ETH/USDC
    capital_total = get_spot_balance()
    grid_levels = 3
    grid_spacing = 10  # 10 USDC d'écart
    allocation_per_grid = 0.2  # 20% du capital par grille
    
    print(f"Solde SPOT disponible: {capital_total:.2f} USDC")
    
    if capital_total < 50:
        print("Fonds insuffisants. Minimum 50 USDC requis.")
        return
    
    # Calcul des paramètres
    price = float(requests.get(f"{BASE_URL}/spotPx?coin=ETH").json()["spotPx"])
    usdc_per_order = capital_total * allocation_per_grid
    eth_per_order = usdc_per_order / price
    
    print(f"Prix actuel: {price:.2f} USDC")
    print(f"Quantité par ordre: {eth_per_order:.4f} ETH")
    
    # Placement des ordres
    for i in range(1, grid_levels + 1):
        buy_price = round(price - (i * grid_spacing), 2)
        sell_price = round(price + (i * grid_spacing), 2)
        
        # Ordre d'achat
        if buy_price > 0:
            print(f"Placing BUY order @ {buy_price} USDC")
            print(place_spot_order("B", "ETH", eth_per_order, buy_price))
        
        # Ordre de vente
        print(f"Placing SELL order @ {sell_price} USDC") 
        print(place_spot_order("S", "ETH", eth_per_order, sell_price))
        
if __name__ == "__main__":
    grid_trading_bot()
