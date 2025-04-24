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
capital_total = 80         # Capital total à allouer (en USDC)
grid_levels = 3             # Nombre de niveaux de la grille
grid_spacing = 10           # Ecart de prix entre chaque niveau (en USDC)
symbol = "ETH/USDC:USDC"    # Paire à trader (SPOT)
pourcentage_par_grille = 0.15   # % du capital à allouer à chaque grille (ex: 0.15 = 15%)

min_order_value = 10        # Minimum requis par Hyperliquid (en USDC)

def get_mid_price(symbol):
    """Récupère le prix médian du marché"""
    markets = dex.load_markets()
    return float(markets[symbol]["info"]["midPx"])

def place_grid_orders():
    base_price = get_mid_price(symbol)
    print(f"Prix de référence : {base_price:.2f} USDC")

    # Calcul du montant par grille selon le pourcentage choisi
    usdc_par_ordre = capital_total * pourcentage_par_grille
    if usdc_par_ordre < min_order_value:
        print(f"ATTENTION : Le montant par grille ({usdc_par_ordre:.2f}$) est inférieur au minimum requis ({min_order_value}$)")
        print("Augmente le capital, le pourcentage, ou diminue le nombre de grilles.")
        return

    amount = usdc_par_ordre / base_price  # Quantité d'ETH par ordre

    print(f"Capital total utilisé : {capital_total} USDC")
    print(f"Nombre de niveaux : {grid_levels}")
    print(f"Pourcentage par grille : {pourcentage_par_grille*100:.1f}%")
    print(f"Montant par grille : {usdc_par_ordre:.2f} USDC")
    print(f"Quantité par ordre : {amount:.6f} ETH")

    for i in range(1, grid_levels + 1):
        buy_price = round(base_price - i * grid_spacing, 2)
        sell_price = round(base_price + i * grid_spacing, 2)

        # Vérifier la valeur de l'ordre
        if (amount * buy_price) >= min_order_value:
            order_buy = dex.create_order(
                symbol=symbol,
                type="limit",
                side="buy",
                amount=amount,
                price=buy_price
            )
            print(f"Ordre BUY placé à {buy_price} : {order_buy['id']}")
        else:
            print(f"Ordre BUY à {buy_price} ignoré (valeur < 10$)")

        if (amount * sell_price) >= min_order_value:
            order_sell = dex.create_order(
                symbol=symbol,
                type="limit",
                side="sell",
                amount=amount,
                price=sell_price
            )
            print(f"Ordre SELL placé à {sell_price} : {order_sell['id']}")
        else:
            print(f"Ordre SELL à {sell_price} ignoré (valeur < 10$)")

if __name__ == "__main__":
    # Vérification du solde
    balance = dex.fetch_balance()
    print("Solde USDC :", balance['USDC']['free'])

    # Placement des ordres grid
    place_grid_orders()
