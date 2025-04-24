import os
from dotenv import load_dotenv
import ccxt

# Charger les variables d'environnement
load_dotenv()
wallet_address = os.getenv("HL_WALLET_ADDRESS")
private_key = os.getenv("HL_PRIVATE_KEY")

# Initialiser la connexion à Hyperliquid via ccxt (SPOT)
dex = ccxt.hyperliquid({
    "walletAddress": wallet_address,
    "privateKey": private_key,
})

# === Paramètres utilisateur ===
capital_total = 89              # Capital total à allouer (en USDC)
grid_levels = 3                 # Nombre de niveaux de la grille (nombre d'ordres buy + sell = grid_levels*2)
grid_spacing = 10               # Ecart de prix entre chaque niveau (en USDC)
symbol = "ETH/USDC:USDC"        # Paire à trader (SPOT)
pourcentage_par_grille = 0.15   # % du capital à allouer à chaque grille (ex: 0.15 = 15%)
min_order_value = 10            # Minimum requis par Hyperliquid (en USDC)

def get_mid_price(symbol):
    """Récupère le prix médian du marché"""
    markets = dex.load_markets()
    return float(markets[symbol]["info"]["midPx"])

def get_usdc_balance(balance):
    """Détecte le solde USDC dans la structure retournée par ccxt"""
    # Essaie les structures classiques
    if 'free' in balance and 'USDC' in balance['free']:
        return float(balance['free']['USDC'])
    elif 'total' in balance and 'USDC' in balance['total']:
        return float(balance['total']['USDC'])
    elif 'USDC' in balance and isinstance(balance['USDC'], dict) and 'free' in balance['USDC']:
        return float(balance['USDC']['free'])
    elif 'USDC' in balance and isinstance(balance['USDC'], (int, float)):
        return float(balance['USDC'])
    else:
        print("\n[DEBUG] Impossible de trouver le solde USDC dans la réponse Hyperliquid :")
        print(balance)
        print("[/DEBUG]\n")
        return 0.0

def place_grid_orders():
    base_price = get_mid_price(symbol)
    print(f"\nPrix de référence : {base_price:.2f} USDC")

    # Calcul du montant par grille selon le pourcentage choisi
    usdc_par_ordre = capital_total * pourcentage_par_grille
    if usdc_par_ordre < min_order_value:
        print(f"ATTENTION : Le montant par grille ({usdc_par_ordre:.2f}$) est inférieur au minimum requis ({min_order_value}$)")
        print("Augmente le capital, le pourcentage, ou diminue le nombre de grilles.")
        return

    amount = usdc_par_ordre / base_price  # Quantité d'ETH par ordre

    # Vérifier le solde disponible
    balance = dex.fetch_balance()
    print("\n[DEBUG] Structure complète du solde retourné par fetch_balance() :")
    print(balance)
    print("[/DEBUG]\n")
    usdc_dispo = get_usdc_balance(balance)
    nombre_ordres = grid_levels * 2
    capital_requis = usdc_par_ordre * nombre_ordres

    print(f"Solde USDC disponible : {usdc_dispo:.2f}")
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

    for i in range(1, grid_levels + 1):
        buy_price = round(base_price - i * grid_spacing, 2)
        sell_price = round(base_price + i * grid_spacing, 2)

        # Vérifier la valeur de l'ordre
        if (amount * buy_price) >= min_order_value:
            try:
                order_buy = dex.create_order(
                    symbol=symbol,
                    type="limit",
                    side="buy",
                    amount=amount,
                    price=buy_price
                )
                print(f"Ordre BUY placé à {buy_price} : {order_buy['id']}")
            except Exception as e:
                print(f"Erreur ordre BUY à {buy_price}: {e}")
        else:
            print(f"Ordre BUY à {buy_price} ignoré (valeur < 10$)")

        if (amount * sell_price) >= min_order_value:
            try:
                order_sell = dex.create_order(
                    symbol=symbol,
                    type="limit",
                    side="sell",
                    amount=amount,
                    price=sell_price
                )
                print(f"Ordre SELL placé à {sell_price} : {order_sell['id']}")
            except Exception as e:
                print(f"Erreur ordre SELL à {sell_price}: {e}")
        else:
            print(f"Ordre SELL à {sell_price} ignoré (valeur < 10$)")

if __name__ == "__main__":
    place_grid_orders()
