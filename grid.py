# hypergrid_bot.py
import os
import time
import requests
import numpy as np
from dotenv import load_dotenv
from scipy.signal import savgol_filter
from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

class HyperliquidGridTrader:
    def __init__(self):
        # Configuration initiale
        self.base_url = "https://api.hyperliquid.xyz"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-KEY": os.getenv("HYPERLIQUID_API_KEY"),
            "X-SECRET": os.getenv("HYPERLIQUID_SECRET")
        }
        
        # Configuration Telegram
        self.tg_bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        # Paramètres de trading
        self.grid_levels = int(os.getenv("GRID_LEVELS", 5))
        self.check_interval = int(os.getenv("ANALYSIS_INTERVAL", 3600))
        self.current_token = None
        self.active_orders = []
        
        # Initialisation du scheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

    # Méthodes d'analyse technique
    def calculate_volatility_score(self, price_data):
        """Calcule le score de volatilité avec lissage Savitzky-Golay"""
        try:
            filtered = savgol_filter(price_data, 15, 3)
            residuals = (price_data - filtered) / filtered
            return np.std(residuals) * 100
        except Exception as e:
            self.send_alert(f"Erreur analyse volatilité : {str(e)}")
            return 0

    def fetch_market_data(self, token):
        """Récupère les données historiques de prix"""
        try:
            response = requests.post(
                f"{self.base_url}/history",
                json={
                    "type": "candle",
                    "coin": token,
                    "interval": "1h",
                    "limit": 100
                },
                headers=self.headers
            )
            return [float(entry[4]) for entry in response.json()]  # Prix de clôture
        except Exception as e:
            self.send_alert(f"Erreur récupération données {token} : {str(e)}")
            return []

    # Sélection de tokens
    def evaluate_tokens(self):
        """Évalue tous les tokens disponibles"""
        try:
            response = requests.post(
                f"{self.base_url}/info",
                json={"type": "meta"},
                headers=self.headers
            )
            tokens = [item["name"] for item in response.json()["universe"]]
            
            scores = {}
            for token in tokens:
                prices = self.fetch_market_data(token)
                if len(prices) > 20:  # Minimum 20 points de données
                    scores[token] = self.calculate_volatility_score(prices[-50:])
            
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        except Exception as e:
            self.send_alert(f"Erreur évaluation tokens : {str(e)}")
            return []

    # Méthodes de trading
    def place_order(self, token, side, price):
        """Place un ordre sur Hyperliquid"""
        order = {
            "coin": token,
            "isBuy": side.lower() == "buy",
            "sz": self.calculate_position_size(token),
            "limitPx": round(price, 4),
            "orderType": "Limit"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/order",
                json=order,
                headers=self.headers
            )
            if response.status_code == 200:
                self.active_orders.append(response.json()["status"]["orderId"])
                return True
            return False
        except Exception as e:
            self.send_alert(f"Erreur ordre {side} {token} : {str(e)}")
            return False

    def calculate_position_size(self, token):
        """Calcule la taille de position selon le capital disponible"""
        try:
            response = requests.post(
                f"{self.base_url}/user",
                json={"user": os.getenv("HYPERLIQUID_API_KEY")},
                headers=self.headers
            )
            balance = float(response.json()["marginSummary"]["accountValue"])
            return round(balance * 0.01, 4)  # 1% du capital par orddre
        except:
            return 0.01  # Fallback

    # Gestion du grid
    def setup_grid(self, token):
        """Initialise la grille de trading"""
        try:
            price = self.fetch_market_data(token)[-1]
            spread = price * 0.005  # 0.5% entre les niveaux
            
            # Annule les anciens ordres
            self.cancel_all_orders()
            
            # Place les ordres de grille
            for i in range(1, self.grid_levels + 1):
                self.place_order(token, "buy", price - (i * spread))
                self.place_order(token, "sell", price + (i * spread))
            
            self.current_token = token
            self.send_alert(f"🔄 Grille initialisée sur {token} | Prix: {price:.4f}")
        except Exception as e:
            self.send_alert(f"Erreur initialisation grille : {str(e)}")

    def cancel_all_orders(self):
        """Annule tous les ordres en cours"""
        try:
            for order_id in self.active_orders:
                requests.delete(
                    f"{self.base_url}/order?orderId={order_id}",
                    headers=self.headers
                )
            self.active_orders = []
        except Exception as e:
            self.send_alert(f"Erreur annulation ordres : {str(e)}")

    # Surveillance périodique
    def periodic_check(self):
        """Vérification régulière du meilleur token"""
        try:
            best_token = self.evaluate_tokens()[0][0]
            
            if best_token != self.current_token:
                self.send_alert(f"🔀 Changement de token : {self.current_token} → {best_token}")
                self.setup_grid(best_token)
        except Exception as e:
            self.send_alert(f"Erreur vérification périodique : {str(e)}")

    # Interface Telegram
    def send_alert(self, message):
        """Envoie une notification Telegram"""
        try:
            self.tg_bot.send_message(
                chat_id=self.chat_id,
                text=f"🚨 HyperGrid Bot:\n{message}"
            )
        except Exception as e:
            print(f"Erreur envoi Telegram: {str(e)}")

    def start_bot(self):
        """Lance le bot"""
        self.send_alert("✅ Bot démarré")
        self.periodic_check()
        self.scheduler.add_job(
            self.periodic_check,
            'interval',
            seconds=self.check_interval
        )

# Commandes Telegram
def tg_command_start(update: Update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Bot HyperGrid Trading actif!\nCommandes disponibles:\n/status - État actuel\n/stop - Arrêter le bot"
    )

if __name__ == "__main__":
    # Initialisation
    trader = HyperliquidGridTrader()
    
    # Configuration Telegram
    updater = Updater(token=os.getenv("TELEGRAM_TOKEN"), use_context=True)
    updater.dispatcher.add_handler(CommandHandler('start', tg_command_start))
    updater.start_polling()
    
    # Démarrage du trading
    trader.start_bot()
    
    # Maintien en vie
    while True:
        time.sleep(1)
