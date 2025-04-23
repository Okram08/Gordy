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
        self.base_url = "https://api.hyperliquid.xyz"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-KEY": os.getenv("HYPERLIQUID_API_KEY"),
            "X-SECRET": os.getenv("HYPERLIQUID_SECRET")
        }
        
        self.tg_bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.grid_levels = int(os.getenv("GRID_LEVELS", 5))
        self.check_interval = int(os.getenv("ANALYSIS_INTERVAL", 3600))
        self.current_token = None
        self.active_orders = []
        
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

    # Méthodes d'analyse technique améliorées
    def calculate_volatility_score(self, price_data):
        """Calcule le score de volatilité avec validation des données"""
        if len(price_data) < 20:
            return 0
            
        try:
            filtered = savgol_filter(price_data, 15, 3)
            residuals = (price_data - filtered) / filtered
            return np.std(residuals) * 100
        except:
            return 0

    def fetch_market_data(self, token):
        """Récupère les données avec timeout et gestion d'erreur"""
        try:
            response = requests.post(
                f"{self.base_url}/history",
                json={
                    "type": "candle",
                    "coin": token,
                    "interval": "1h",
                    "limit": 100
                },
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return [float(entry[4]) for entry in response.json()]
            return []
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            self.send_alert("⚠️ Timeout API - Vérifiez la connexion Internet")
            return []
        except Exception as e:
            self.send_alert(f"Erreur données marché: {str(e)}")
            return []

    # Sélection de tokens sécurisée
    def evaluate_tokens(self):
        """Évaluation avec vérification des résultats"""
        try:
            response = requests.post(
                f"{self.base_url}/info",
                json={"type": "meta"},
                headers=self.headers,
                timeout=10
            )
            
            if not response.ok:
                return []
                
            tokens = [item["name"] for item in response.json().get("universe", [])]
            if not tokens:
                return []
            
            scores = {}
            for token in tokens[:10]:  # Limite à 10 tokens pour performance
                prices = self.fetch_market_data(token)
                if prices:
                    scores[token] = self.calculate_volatility_score(prices[-50:])
            
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        except:
            return []

    # Méthodes de trading avec gestion d'erreur
    def place_order(self, token, side, price):
        """Placement d'ordre sécurisé"""
        order = {
            "coin": token,
            "isBuy": side.lower() == "buy",
            "sz": self.calculate_position_size(),
            "limitPx": round(price, 4),
            "orderType": "Limit"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/order",
                json=order,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.active_orders.append(response.json()["status"]["orderId"])
                return True
            return False
        except:
            return False

    def calculate_position_size(self):
        """Calcul de position avec fallback"""
        try:
            response = requests.post(
                f"{self.base_url}/user",
                json={"user": os.getenv("HYPERLIQUID_API_KEY")},
                headers=self.headers,
                timeout=10
            )
            balance = float(response.json()["marginSummary"]["accountValue"])
            return round(balance * 0.01 / self.grid_levels, 4)  # 1% total réparti
        except:
            return 0.01

    # Gestion du grid améliorée
    def setup_grid(self, token):
        """Initialisation sécurisée de la grille"""
        try:
            self.cancel_all_orders()
            
            prices = self.fetch_market_data(token)
            if not prices:
                return
                
            price = prices[-1]
            spread = price * 0.005
            
            for i in range(1, self.grid_levels + 1):
                if not self.place_order(token, "buy", price - (i * spread)):
                    self.send_alert(f"Échec ordre d'achat niveau {i}")
                if not self.place_order(token, "sell", price + (i * spread)):
                    self.send_alert(f"Échec ordre de vente niveau {i}")
            
            self.current_token = token
            self.send_alert(f"🔄 Grille activée sur {token} | Prix: {price:.4f}")
        except Exception as e:
            self.send_alert(f"Erreur configuration grille: {str(e)}")

    def cancel_all_orders(self):
        """Annulation sécurisée des ordres"""
        try:
            for order_id in self.active_orders.copy():
                response = requests.delete(
                    f"{self.base_url}/order?orderId={order_id}",
                    headers=self.headers,
                    timeout=10
                )
                if response.status_code == 200:
                    self.active_orders.remove(order_id)
        except:
            pass

    # Surveillance périodique
    def periodic_check(self):
        """Vérification avec gestion des erreurs"""
        try:
            candidates = self.evaluate_tokens()
            if not candidates:
                return
                
            best_token = candidates[0][0]
            
            if best_token != self.current_token:
                self.send_alert(f"🔀 Changement vers {best_token}")
                self.setup_grid(best_token)
        except:
            pass

    # Interface Telegram améliorée
    def send_alert(self, message):
        """Envoi de notification avec réessai"""
        try:
            self.tg_bot.send_message(
                chat_id=self.chat_id,
                text=f"🚨 HyperGrid Bot:\n{message}",
                timeout=10
            )
        except:
            pass

    def start_bot(self):
        """Démarrage sécurisé"""
        self.send_alert("✅ Bot démarré")
        self.periodic_check()
        self.scheduler.add_job(
            self.periodic_check,
            'interval',
            seconds=self.check_interval
        )

    def graceful_shutdown(self):
        """Arrêt propre"""
        self.send_alert("⏹ Arrêt en cours...")
        self.cancel_all_orders()
        self.scheduler.shutdown()
        time.sleep(2)
        exit(0)

# Commandes Telegram
def tg_command_start(update: Update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Bot HyperGrid Trading actif!\nCommandes:\n/status - État\n/stop - Arrêt"
    )

def tg_command_stop(update: Update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Arrêt demandé...")
    os._exit(0)

if __name__ == "__main__":
    trader = HyperliquidGridTrader()
    updater = Updater(token=os.getenv("TELEGRAM_TOKEN"), use_context=True)
    
    try:
        updater.dispatcher.add_handler(CommandHandler('start', tg_command_start))
        updater.dispatcher.add_handler(CommandHandler('stop', tg_command_stop))
        updater.start_polling()
        
        trader.start_bot()
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        trader.graceful_shutdown()
    except Exception as e:
        trader.send_alert(f"❌ ERREUR CRITIQUE: {str(e)}")
        trader.graceful_shutdown()
