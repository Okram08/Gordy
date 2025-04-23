import ccxt
import logging
import time
from dotenv import load_dotenv
import os
import random

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Récupérer les clés API depuis les variables d'environnement
api_key = os.getenv('HYPERLIQUID_API_KEY')
private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')

if not api_key or not private_key:
    raise ValueError("Les clés API ou privée ne sont pas chargées correctement depuis .env")

# Connexion à l'API de Hyperliquid via ccxt
exchange = ccxt.hyperliquid({
    'apiKey': api_key,
    'privateKey': private_key,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

class GridBot:
    def __init__(self, exchange):
        self.exchange = exchange
        self.symbols = []
        self.grid_upper = 0.15  # Prix supérieur de la grille (en pourcentage)
        self.grid_lower = 0.10  # Prix inférieur de la grille (en pourcentage)
        self.order_amount = 10  # Quantité de crypto à acheter/vendre par ordre

    def fetch_symbols(self):
        """Récupérer les symboles des crypto-monnaies disponibles sur l'échange"""
        markets = self.exchange.load_markets()
        self.symbols = [symbol for symbol in markets if '/USDC' in symbol]
        logger.info(f"Symboles récupérés : {self.symbols}")

    def analyze_market(self, symbol):
        """Analyser les crypto-monnaies pour sélectionner les plus propices au grid trading"""
        # Pour cet exemple, on va choisir un critère simple: la volatilité
        # On récupère les données de marché et on vérifie la volatilité (ex. écart-type sur les 24h)
        ticker = self.exchange.fetch_ticker(symbol)
        price_change = ticker['percentage']  # Changement en pourcentage sur 24h
        if abs(price_change) > 5:  # Choisir les cryptos avec une volatilité supérieure à 5%
            return True
        return False

    def place_orders(self, symbol, price):
        """Placer des ordres de grid trading"""
        logger.info(f"Placer des ordres pour {symbol} à {price:.2f}")

        # Placer un ordre d'achat à un prix inférieur
        buy_price = price * (1 - self.grid_lower)
        logger.info(f"Placer un ordre d'achat à {buy_price:.2f}")
        self.exchange.create_limit_buy_order(symbol, self.order_amount, buy_price)

        # Placer un ordre de vente à un prix supérieur
        sell_price = price * (1 + self.grid_upper)
        logger.info(f"Placer un ordre de vente à {sell_price:.2f}")
        self.exchange.create_limit_sell_order(symbol, self.order_amount, sell_price)

    def run(self):
        """Lancer le bot de grid trading"""
        logger.info("Scheduler started")
        self.fetch_symbols()

        for symbol in self.symbols:
            if self.analyze_market(symbol):
                logger.info(f"Analyse réussie pour {symbol}")
                ticker = self.exchange.fetch_ticker(symbol)
                last_price = ticker['last']
                logger.info(f"Vérification du symbole {symbol} à {last_price:.2f}")
                self.place_orders(symbol, last_price)
            else:
                logger.info(f"{symbol} n'est pas propice pour le grid trading en ce moment.")
            
            # Attendre un certain temps avant de passer à la prochaine crypto
            time.sleep(5)

# Lancer le bot
if __name__ == "__main__":
    bot = GridBot(exchange)
    bot.run()
