import os
import time
from dotenv import load_dotenv
import ccxt

# Charger les variables d'environnement
load_dotenv()
wallet_address = os.getenv("HL_WALLET_ADDRESS")
private_key = os.getenv("HL_PRIVATE_KEY")

# Initialiser la connexion à Hyperliquid via ccxt
dex = ccxt.hyperliquid({
    "walletAddress": wallet_address,
    "privateKey": private_key,
})

# === Paramètres utilisateur ===
capital_total = 50         # Capital total à allouer (en USDC)
grid_levels = 5             # Nombre de niveaux de la grille
grid_spacing = 10           # Ecart de prix entre chaque niveau (en USDC)
symbol = "ETH/USDC:USDC"    # Paire à trader

def get_mid_price(symbol):
    """Récupère le prix médian du marché"""
    markets = dex.load_markets()
    return float(markets[symbol]["info"]["midPx"])

def calc_grid(base_price):
    """Calcule les niveaux de la grille"""
    buy_prices = [round(base_price - i * grid_spacing, 2) for i in range(1, grid_levels + 1)]
    sell_prices = [round(base_price + i * grid_spacing, 2) for i in range(1, grid_levels + 1)]
    return buy_prices, sell_prices

def get_open_orders():
    """Récupère les ordres ouverts pour la paire"""
    orders = dex.fetch_open_orders(symbol)
    return orders

def cancel_order(order_id):
    """Annule un ordre"""
    try:
        dex.cancel_order(order_id, symbol)
        print(f"Ordre annulé : {order_id}")
    except Exception as e:
        print(f"Erreur lors de l'annulation de l'ordre {order_id} : {e}")

def place_order(side, price, amount):
    """Place un ordre limite"""
    try:
        order = dex.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=amount,
            price=price
        )
        print(f"Ordre {side.upper()} placé à {price} : {order['id']}")
        return order
    except Exception as e:
        print(f"Erreur lors du placement de l'ordre {side} à {price} : {e}")
        return None

def main_loop():
    print("=== Grid trading bot Hyperliquid ===")
    while True:
        try:
            base_price = get_mid_price(symbol)
            buy_prices, sell_prices = calc_grid(base_price)
            nombre_ordres = grid_levels * 2
            usdc_par_ordre = capital_total / nombre_ordres
            amount = usdc_par_ordre / base_price  # Quantité d'ETH par ordre

            # Récupérer les ordres ouverts
            open_orders = get_open_orders()
            open_buys = [float(o['price']) for o in open_orders if o['side'] == 'buy']
            open_sells = [float(o['price']) for o in open_orders if o['side'] == 'sell']

            # Placer les ordres manquants
            for price in buy_prices:
                if price not in open_buys:
                    place_order('buy', price, amount)
            for price in sell_prices:
                if price not in open_sells:
                    place_order('sell', price, amount)

            # Afficher l'état de la grille
            print(f"\n[{time.strftime('%H:%M:%S')}] Grille autour de {base_price:.2f} USDC")
            print("Ordres BUY ouverts :", sorted(open_buys))
            print("Ordres SELL ouverts :", sorted(open_sells))
            print("Attente 60 secondes...\n")
            time.sleep(60)
        except Exception as e:
            print("Erreur dans la boucle principale :", e)
            time.sleep(60)

if __name__ == "__main__":
    balance = dex.fetch_balance()
    print("Solde USDC :", balance['USDC']['free'])
    main_loop()
