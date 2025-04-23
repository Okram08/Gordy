import os
import time
import asyncio
import requests
import numpy as np
import pytz
import logging
from dotenv import load_dotenv
from scipy.signal import savgol_filter
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import utc

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        self.check_interval = int(os.getenv("ANALYSIS_INTERVAL", 60))  # 1min pour debug
        self.current_token = None
        self.active_orders = []
        self.scheduler = BackgroundScheduler(timezone=utc)
        self.scheduler.start()

    def calculate_volatility_score(self, price_data):
        if len(price_data) < 5:  # assoupli pour debug
            logger.info(f"Pas assez de données pour calculer la volatilité ({len(price_data)} points)")
            return 0
        try:
            filtered = savgol_filter(price_data, 5, 2)  # fenêtre plus petite pour debug
            residuals = (price_data - filtered) / filtered
            score = np.std(residuals) * 100
            logger.info(f"Score volatilité calculé: {score:.4f}")
            return score
        except Exception as e:
            logger.error(f"Erreur calcul volatilité: {str(e)}")
            return 0

    def fetch_market_data(self, token):
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
                prices = [float(entry[4]) for entry in response.json()]
                logger.info(f"{token}: prix récupérés (premiers 5): {prices[:5]}")
                return prices
            else:
                logger.warning(f"{token}: pas de données de prix (status {response.status_code})")
            return []
        except Exception as e:
            logger.error(f"Erreur récupération données {token}: {str(e)}")
            return []

    def evaluate_tokens(self):
        try:
            response = requests.post(
                f"{self.base_url}/info",
                json={"type": "meta"},
                headers=self.headers,
                timeout=10
            )
            data = response.json()
            tokens = [item["name"] for item in data.get("universe", [])]
            logger.info(f"Tokens récupérés: {tokens}")
            if not tokens:
                logger.warning("Aucun token trouvé dans la réponse API")
                return []
            scores = {}
            for token in tokens[:10]:
                prices = self.fetch_market_data(token)
                if prices:
                    score = self.calculate_volatility_score(prices[-20:])
                    scores[token] = score
                else:
                    logger.info(f"{token}: pas de prix récupérés")
            logger.info(f"Scores de volatilité: {scores}")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_scores:
                logger.info(f"Meilleurs tokens: {sorted_scores[:3]}")
            else:
                logger.info("Aucun token n'a passé le filtre de volatilité")
            return sorted_scores[:3] or []
        except Exception as e:
            logger.error(f"Erreur évaluation tokens: {str(e)}")
            return []

    def place_order(self, token, side, price):
        try:
            order = {
                "coin": token,
                "isBuy": side.lower() == "buy",
                "sz": self.calculate_position_size(),
                "limitPx": round(price, 4),
                "orderType": "Limit"
            }
            logger.info(f"Placement d'ordre: {order}")
            response = requests.post(
                f"{self.base_url}/order",
                json=order,
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                self.active_orders.append(response.json()["status"]["orderId"])
                logger.info(f"Ordre {side} placé pour {token} à {price}")
                return True
            else:
                logger.warning(f"Echec ordre {side} {token} (status {response.status_code})")
            return False
        except Exception as e:
            logger.error(f"Erreur ordre {side}: {str(e)}")
            return False

    def calculate_position_size(self):
        try:
            response = requests.post(
                f"{self.base_url}/user",
                json={"user": os.getenv("HYPERLIQUID_API_KEY")},
                headers=self.headers,
                timeout=10
            )
            balance = float(response.json()["marginSummary"]["accountValue"])
            size = round(balance * 0.01 / self.grid_levels, 4)
            logger.info(f"Taille de position calculée: {size}")
            return size
        except Exception as e:
            logger.error(f"Erreur calcul taille position: {str(e)}")
            return 0.01

    def setup_grid(self, token):
        try:
            self.cancel_all_orders()
            prices = self.fetch_market_data(token)
            if not prices:
                logger.warning(f"Pas de prix pour {token}, grille non créée.")
                return
            price = prices[-1]
            spread = price * 0.005
            for i in range(1, self.grid_levels + 1):
                self.place_order(token, "buy", price - (i * spread))
                self.place_order(token, "sell", price + (i * spread))
            self.current_token = token
            self.send_alert(f"🔄 Grille activée sur {token} | Prix: {price:.4f}")
        except Exception as e:
            logger.error(f"Erreur configuration grille: {str(e)}")

    def cancel_all_orders(self):
        try:
            for order_id in self.active_orders.copy():
                response = requests.delete(
                    f"{self.base_url}/order?orderId={order_id}",
                    headers=self.headers,
                    timeout=10
                )
                if response.status_code == 200:
                    self.active_orders.remove(order_id)
        except Exception as e:
            logger.error(f"Erreur annulation ordres: {str(e)}")

    def periodic_check(self):
        try:
            candidates = self.evaluate_tokens()
            if not candidates:
                logger.info("Aucun token éligible trouvé à ce cycle.")
                return
            best_token = candidates[0][0]
            logger.info(f"Meilleur token détecté: {best_token}")
            if best_token != self.current_token:
                self.send_alert(f"🔀 Changement vers {best_token}")
                self.setup_grid(best_token)
        except Exception as e:
            logger.error(f"Erreur vérification périodique: {str(e)}")

    def send_alert(self, message):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.tg_bot.send_message(
                    chat_id=self.chat_id,
                    text=f"🚨 HyperGrid Bot:\n{message}"
                )
            )
        except Exception as e:
            logger.error(f"Erreur envoi Telegram: {str(e)}")

    def start_bot(self):
        self.send_alert("✅ Bot démarré")
        self.periodic_check()
        self.scheduler.add_job(
            self.periodic_check,
            'interval',
            seconds=self.check_interval
        )
        logger.info("Bot complètement initialisé")

    def graceful_shutdown(self):
        self.send_alert("⏹ Arrêt en cours...")
        self.cancel_all_orders()
        self.scheduler.shutdown()
        time.sleep(2)
        exit(0)

# Commandes Telegram
async def tg_command_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 HyperGrid Trading Bot Actif\n"
        "Commandes disponibles:\n"
        "/status - État actuel\n"
        "/stop - Arrêter le bot"
    )

async def tg_command_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Arrêt demandé...")
    os._exit(0)

if __name__ == "__main__":
    trader = HyperliquidGridTrader()
    application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()
    application.add_handler(CommandHandler('start', tg_command_start))
    application.add_handler(CommandHandler('stop', tg_command_stop))
    import threading
    trading_thread = threading.Thread(target=trader.start_bot, daemon=True)
    trading_thread.start()
    try:
        application.run_polling()
    except KeyboardInterrupt:
        trader.graceful_shutdown()
    except Exception as e:
        logger.critical(f"ERREUR FATALE: {str(e)}")
        trader.graceful_shutdown()
