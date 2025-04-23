import os
import time
import asyncio
import requests
import logging
from dotenv import load_dotenv
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import utc

# --- CONFIGURATION ---
load_dotenv()
API_URL = "https://api.hyperliquid.xyz"
HEADERS = {"Content-Type": "application/json"}
GRID_LEVELS = int(os.getenv("GRID_LEVELS", 5))
GRID_STEP_PCT = float(os.getenv("GRID_STEP_PCT", 0.005))  # 0.5% par défaut
CHECK_INTERVAL = int(os.getenv("ANALYSIS_INTERVAL", 60))  # 60s par défaut

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FONCTIONS API HYPERLIQUID ---

def get_spot_meta_and_prices():
    resp = requests.post(f"{API_URL}/info", json={"type": "spotMetaAndAssetCtxs"}, headers=HEADERS)
    data = resp.json()
    spot_meta = data[0]["universe"]  # Liste des paires spot
    spot_ctxs = data[1]              # Liste des contextes, avec prix
    # Associe chaque paire à son contexte (prix, volume, etc.)
    meta_by_index = {pair["index"]: pair for pair in spot_meta}
    ctx_by_index = {ctx["a"]: ctx for ctx in spot_ctxs}
    pairs = []
    for idx, pair in meta_by_index.items():
        ctx = ctx_by_index.get(idx)
        if ctx and "markPx" in ctx:
            pair_info = {
                "name": pair["name"],
                "index": idx,
                "price": float(ctx["markPx"]),
                "volume": float(ctx.get("dayNtlVlm", 0))
            }
            pairs.append(pair_info)
    return pairs

def get_asset_id(spot_index):
    return 10000 + spot_index

def build_spot_order_payload(asset_id, is_buy, price, size, api_key, signature, nonce):
    payload = {
        "action": {
            "type": "order",
            "orders": [{
                "a": asset_id,
                "b": is_buy,
                "p": str(price),
                "s": str(size),
                "r": False,
                "t": {"limit": {"tif": "Gtc"}}
            }]
        },
        "nonce": nonce,
        "signature": signature,      # À générer avec le SDK officiel
        "X-API-KEY": api_key
    }
    return payload

def place_spot_order(payload):
    resp = requests.post(f"{API_URL}/exchange", json=payload, headers=HEADERS)
    return resp.json()

# --- BOT GRID TRADING ---

class HyperliquidSpotGridBot:
    def __init__(self):
        self.tg_bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.scheduler = BackgroundScheduler(timezone=utc)
        self.scheduler.start()
        self.selected_pair = None
        self.grid_orders = []
        self.running = True

    def send_alert(self, message):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.tg_bot.send_message(
                    chat_id=self.chat_id,
                    text=f"🚨 HyperGrid Spot Bot:\n{message}"
                )
            )
        except Exception as e:
            logger.error(f"Erreur envoi Telegram: {str(e)}")

    def select_best_pair(self):
        pairs = get_spot_meta_and_prices()
        logger.info(f"Paires spot trouvées: {[p['name'] for p in pairs]}")
        # Critère simple : volume le plus élevé
        best = None
        best_vol = 0
        for pair in pairs:
            logger.info(f"{pair['name']} : volume 24h = {pair['volume']}, prix = {pair['price']}")
            if pair["volume"] > best_vol:
                best = pair
                best_vol = pair["volume"]
        if best:
            logger.info(f"Meilleure paire spot sélectionnée: {best['name']}")
        else:
            logger.warning("Aucune paire spot trouvée avec volume et prix.")
        return best

    def setup_grid_orders(self, pair):
        price = pair["price"]
        grid = []
        for i in range(1, GRID_LEVELS + 1):
            buy_price = round(price * (1 - GRID_STEP_PCT * i), 6)
            sell_price = round(price * (1 + GRID_STEP_PCT * i), 6)
            grid.append(("buy", buy_price))
            grid.append(("sell", sell_price))
        logger.info(f"Grille générée pour {pair['name']} : {grid}")
        return grid

    def periodic_check(self):
        if not self.running:
            return
        pair = self.select_best_pair()
        if not pair:
            logger.warning("Aucune paire spot sélectionnable à ce cycle.")
            return
        if not self.selected_pair or pair["name"] != self.selected_pair["name"]:
            self.selected_pair = pair
            self.send_alert(f"Nouvelle paire spot sélectionnée: {pair['name']} (prix {pair['price']})")
        # Génère la grille
        self.grid_orders = self.setup_grid_orders(pair)
        self.send_alert(
            f"Grille SPOT pour {pair['name']} (prix {pair['price']})\n" +
            "\n".join([f"{side.upper()} {p}" for side, p in self.grid_orders])
        )
        # --- EXEMPLE (simulation) ---
        # Pour chaque niveau, affiche le payload à signer pour un ordre
        api_key = os.getenv("HYPERLIQUID_API_KEY")
        # TODO: génère signature avec le SDK officiel
        fake_signature = "signature_a_remplir"
        nonce = int(time.time() * 1000)
        asset_id = get_asset_id(pair["index"])
        size = 1  # À adapter
        for side, px in self.grid_orders:
            payload = build_spot_order_payload(
                asset_id, side == "buy", px, size, api_key, fake_signature, nonce
            )
            logger.info(f"Payload à signer pour {side} {px}: {payload}")
            # Pour passer l'ordre : décommente la ligne suivante après avoir intégré la signature réelle
            # resp = place_spot_order(payload)
            # logger.info(f"Réponse Hyperliquid: {resp}")

    def start_bot(self):
        self.send_alert("✅ Bot SPOT démarré")
        self.periodic_check()
        self.scheduler.add_job(
            self.periodic_check,
            'interval',
            seconds=CHECK_INTERVAL
        )
        logger.info("Bot complètement initialisé")

    def graceful_shutdown(self):
        self.send_alert("⏹ Arrêt en cours...")
        self.scheduler.shutdown()
        time.sleep(2)
        exit(0)

# --- Commandes Telegram ---
async def tg_command_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 HyperGrid Spot Bot Actif\n"
        "Commandes disponibles:\n"
        "/status - État actuel\n"
        "/stop - Arrêter le bot"
    )

async def tg_command_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Arrêt demandé...")
    os._exit(0)

if __name__ == "__main__":
    bot = HyperliquidSpotGridBot()
    application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()
    application.add_handler(CommandHandler('start', tg_command_start))
    application.add_handler(CommandHandler('stop', tg_command_stop))
    import threading
    trading_thread = threading.Thread(target=bot.start_bot, daemon=True)
    trading_thread.start()
    try:
        application.run_polling()
    except KeyboardInterrupt:
        bot.graceful_shutdown()
    except Exception as e:
        logger.critical(f"ERREUR FATALE: {str(e)}")
        bot.graceful_shutdown()
