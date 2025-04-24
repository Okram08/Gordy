import os
import requests
from dotenv import load_dotenv

# Charger l'adresse publique depuis .env
load_dotenv()
wallet_address = os.getenv("HL_WALLET_ADDRESS")

# Paramètres utilisateur
capital_total = 89              # Capital total à allouer (en USDC)
grid_levels = 3                 # Nombre de niveaux de la grille
grid_spacing = 10               # Ecart de prix entre chaque niveau (en USDC)
symbol = "ETH-USDC"             # Paire à trader (SPOT, format natif Hyperliquid)
pourcentage_par_grille = 0.15   # % du capital à allouer à chaque grille (ex: 0.15 = 15%)
min_order_value = 10            # Minimum requis par Hyperliquid (en USDC)

def get_spot_balance(address):
    url = "https://api.hyperliquid.xyz/info"
    payload = {
        "type": "spotUser",
        "user": address
    }
    resp = requests.post(url, json=payload)
    data = resp.json()
    # Structure attendue : {'assetPositions':[{'coin':'USDC','position':'89.0'}, ...]}
    usdc = 0.0
    if "assetPositions" in data:
        for asset in data["assetPositions"]:
            if asset["coin"] == "USDC":
                usdc = float(asset["position"])
    return usdc, data

def get_spot_price(symbol):
    url = "https://api.hyperliquid.xyz/info"
    payload = {
        "type": "spotPx",
        "coin": symbol.split('-')[0]
    }
    resp = requests.post(url, json=payload)
    data = resp.json()
    # Structure attendue : {'spotPx': 1745.65, ...}
    return float(data["spotPx"])

def main():
    # Lire le solde spot USDC
    usdc_dispo, raw_balance = get_spot_balance(wallet_address)
    print(f"\n[DEBUG] Structure complète du solde spot :\n{raw_balance}\n")
    print(f"Solde USDC spot disponible : {usdc_dispo:.2f} USDC")

    # Calcul des ordres grid
    base_price = get_spot_price(symbol)
    print(f"Prix spot de référence : {base_price:.2f} USDC")

    usdc_par_ordre = capital_total * pourcentage_par_grille
    if usdc_par_ordre < min_order_value:
        print(f"ATTENTION : Le montant par grille ({usdc_par_ordre:.2f}$) est inférieur au minimum requis ({min_order_value}$)")
        print("Augmente le capital, le pourcentage, ou diminue le nombre de grilles.")
        return

    amount = usdc_par_ordre / base_price  # Quantité d'ETH par ordre
    nombre_ordres = grid_levels * 2
    capital_requis = usdc_par_ordre * nombre_ordres

    print(f"Capital requis pour la grille : {capital_requis:.2f} USDC")
    if usdc_dispo < capital_requis:
        print("ERREUR : Solde insuffisant pour placer tous les ordres de la grille !")
        print("Dépose plus de fonds ou réduis le capital total / nombre de grilles.")
        return

    print(f"Capital total utilisé : {capital_total} USDC")
    print(f"Nombre de niveaux : {grid_levels}")
    print(f"Pourcentage par grille : {pourcentage_par_grille*100:.1f}%")
    print(f"Montant par grille : {usdc_par_ordre:.2f} USDC")
    print(f"Quantité par ordre : {amount:.6f} ETH\n")

    # Afficher les ordres à placer
    for i in range(1, grid_levels + 1):
        buy_price = round(base_price - i * grid_spacing, 2)
        sell_price = round(base_price + i * grid_spacing, 2)

        if (amount * buy_price) >= min_order_value:
            print(f"[A placer] Ordre BUY à {buy_price} USDC pour {amount:.6f} ETH (~{amount*buy_price:.2f} USDC)")
        else:
            print(f"Ordre BUY à {buy_price} ignoré (valeur < 10$)")

        if (amount * sell_price) >= min_order_value:
            print(f"[A placer] Ordre SELL à {sell_price} USDC pour {amount:.6f} ETH (~{amount*sell_price:.2f} USDC)")
        else:
            print(f"Ordre SELL à {sell_price} ignoré (valeur < 10$)")

if __name__ == "__main__":
    main()
