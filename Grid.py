import os
import logging
from dotenv import load_dotenv
from binance.client import Client

# Charger les variables d'environnement
load_dotenv()

# Configuration des logs
logging.basicConfig(level=logging.INFO)

class GridBot:
    def __init__(self):
        # Charger les clés API à partir du fichier .env
        self.api_key = os.getenv("HYPERLIQUID_API_KEY")
        self.api_secret = os.getenv("HYPERLIQUID_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Les clés API ou privée ne sont pas chargées correctement depuis .env")
        
        self.client = Client(self.api_key, self.api_secret)
        
    def get_market_data(self, symbol):
        # Cette méthode est censée récupérer les données du marché pour un symbole donné.
        # Remplace cette logique par l'API que tu utilises pour obtenir les données en temps réel.
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            if ticker:
                # Supposons que le prix actuel et le prix précédent sont extraits ici
                current_price = float(ticker['price'])
                # Logique pour obtenir le prix précédent (exemple avec une différence d'une minute ou autre)
                previous_price = current_price * 0.95  # Exemple fictif (changer selon ta logique)
                
                return {'current_price': current_price, 'previous_price': previous_price}
            else:
                logging.warning(f"Pas de données disponibles pour {symbol}.")
                return None
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des données pour {symbol}: {str(e)}")
            return None
    
    def analyze_market(self, symbol):
        # Récupérer les données du marché pour le symbole
        price_data = self.get_market_data(symbol)
        
        if price_data is None:
            logging.warning(f"Pas de données pour le symbole {symbol}.")
            return False
        
        # Extraire les prix (par exemple, prix actuel et prix précédent)
        current_price = price_data['current_price']
        previous_price = price_data['previous_price']
        
        # Vérifier que les prix sont valides
        if current_price is None or previous_price is None:
            logging.warning(f"Prix non valides pour {symbol}: current_price={current_price}, previous_price={previous_price}.")
            return False

        # Calculer le changement de prix en pourcentage
        price_change = (current_price - previous_price) / previous_price * 100
        
        # Vérifier si le changement de prix est valide
        if price_change is None:
            logging.warning(f"Changement de prix invalide pour {symbol}: price_change={price_change}.")
            return False
        
        # Analyser la volatilité
        logging.info(f"Changement de prix pour {symbol}: {price_change}%")
        if abs(price_change) > 5:  # Choisir les cryptos avec une volatilité supérieure à 5%
            return True
        return False

    def select_best_markets(self):
        # Exemple de liste de symboles (tu peux l'adapter selon tes besoins)
        symbols = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'SOLUSD', 'ADAUSD', 'DOGEUSD']
        
        selected_symbols = []
        
        for symbol in symbols:
            if self.analyze_market(symbol):
                selected_symbols.append(symbol)
        
        logging.info(f"Symboles sélectionnés pour le Grid Trading: {selected_symbols}")
        return selected_symbols
    
    def run(self):
        logging.info("Scheduler started")
        
        # Sélectionner les symboles les plus volatiles pour le Grid Trading
        selected_symbols = self.select_best_markets()
        
        if not selected_symbols:
            logging.warning("Aucun symbole n'a été sélectionné pour le Grid Trading.")
            return
        
        for symbol in selected_symbols:
            logging.info(f"Démarrage du Grid Trading pour {symbol}")
            # Logique du Grid Trading ici...
            # Par exemple, tu peux utiliser un bot de trading pour initier les ordres de Grid.
            # Exemple : self.start_grid_trading(symbol)
    
if __name__ == "__main__":
    try:
        bot = GridBot()
        bot.run()
    except ValueError as e:
        logging.error(str(e))
