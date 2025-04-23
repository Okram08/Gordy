import os
import time
import logging
from datetime import datetime
import ccxt
from telegram import Bot
from telegram.ext import Updater, CommandHandler
from dotenv import load_dotenv

# Configuration initiale
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HyperliquidAPI:
    def __init__(self, api_key, api_secret):
        self.exchange = ccxt.hyperliquid({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })

    def fetch_symbols(self):
        """Retourne tous les symboles disponibles"""
        try:
            markets = self.exchange.load_markets()
            symbols = [market for market in markets if 'USDT' in market]
            logging.info(f"Symboles récupérés : {symbols}")
            return symbols
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des symboles : {e}")
            return []

    def fetch_ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    def create_limit_buy_order(self, symbol, amount, price):
        """Créer un ordre limite d'achat"""
        return self.exchange.create_order(symbol, 'limit', 'buy', amount, price, params={'timeInForce': 'PostOnly'})

    def create_limit_sell_order(self, symbol, amount, price):
        """Créer un ordre limite de vente"""
        return self.exchange.create_order(symbol, 'limit', 'sell', amount, price, params={'timeInForce': 'PostOnly'})

    def milliseconds(self):
        return self.exchange.milliseconds()

class GridTradingBot:
    def __init__(self, exchange, capital, grid_levels, dry_run, scan_interval):
        self.exchange = exchange
        self.capital = capital
        self.grid_levels = grid_levels
        self.dry_run = dry_run
        self.scan_interval = scan_interval

        self.running = True
        self.symbols = []
        self.current_symbol = None
        self.order_amount = None
        self.grid_lower = None
        self.grid_upper = None

    def fetch_and_filter_symbols(self):
        """Récupère et filtre les symboles disponibles"""
        self.symbols = self.exchange.fetch_symbols()
        logging.info(f"Analyse de {len(self.symbols)} symboles...")
        if not self.symbols:
            logging.warning("Aucun symbole valide trouvé pour l'analyse.")

    def place_orders(self, symbol, price):
        """Placer des ordres d'achat et de vente autour de la grille"""
        grid_range = 0.02 * price
        self.grid_lower = price - grid_range
        self.grid_upper = price + grid_range
        order_amount = self.capital / self.grid_levels / price  # Calcul de la taille des ordres

        logging.info(f"Placer des ordres pour {symbol} à {price:.2f}:")
        logging.info(f"Ordre d'achat à {self.grid_lower:.2f}, Ordre de vente à {self.grid_upper:.2f}")
        
        if not self.dry_run:
            self.exchange.create_limit_buy_order(symbol, order_amount, self.grid_lower)
            self.exchange.create_limit_sell_order(symbol, order_amount, self.grid_upper)
        
    def run(self):
        self.fetch_and_filter_symbols()
        while self.running:
            for symbol in self.symbols:
                self.current_symbol = symbol
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker['last']
                logging.info(f"Vérification du symbole {symbol} à {price:.2f}")

                # Créer et placer les ordres
                self.place_orders(symbol, price)

                time.sleep(self.scan_interval)

class TelegramInterface:
    def __init__(self, token, chat_id, bot_logic):
        self.token = token
        self.chat_id = chat_id
        self.bot = Bot(token=self.token)
        self.updater = Updater(token=self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.bot_logic = bot_logic

        self.dispatcher.add_handler(CommandHandler("status", self.status))
        self.dispatcher.add_handler(CommandHandler("pause", self.pause))
        self.dispatcher.add_handler(CommandHandler("resume", self.resume))
        self.dispatcher.add_handler(CommandHandler("capital", self.set_capital, pass_args=True))
        self.dispatcher.add_handler(CommandHandler("gridreport", self.grid_report))
        self.dispatcher.add_handler(CommandHandler("stop", self.stop))
        self.dispatcher.add_handler(CommandHandler("help", self.help))

    def send_message(self, message):
        self.bot.send_message(chat_id=self.chat_id, text=message)

    def status(self, update, context):
        etat = "✅ Actif" if self.bot_logic.running else "⏸ Pause"
        msg = (
            f"📊 Status Bot\n"
            f"Actif: {self.bot_logic.current_symbol or 'Aucun'}\n"
            f"Capital: {self.bot_logic.capital:.2f} USDT\n"
            f"Etat: {etat}"
        )
        self.send_message(msg)

    def pause(self, update, context):
        self.bot_logic.running = False
        self.send_message("⏸ Bot mis en pause")

    def resume(self, update, context):
        self.bot_logic.running = True
        self.send_message("▶ Reprise des opérations")

    def stop(self, update, context):
        self.send_message("🛑 Arrêt complet du bot...")
        self.bot_logic.running = False
        os._exit(0)

    def help(self, update, context):
        help_msg = (
            "❓ Commandes disponibles:\n"
            "/status - Etat actuel\n"
            "/pause - Mettre en pause\n"
            "/resume - Reprendre\n"
            "/capital <montant> - Modifier capital\n"
            "/gridreport - Détails grille\n"
            "/stop - Arrêt complet\n"
            "/help - Aide"
        )
        self.send_message(help_msg)

    def set_capital(self, update, context):
        try:
            new_capital = float(context.args[0])
            self.bot_logic.capital = new_capital
            self.send_message(f"💰 Capital mis à jour: {new_capital} USDT")
        except:
            self.send_message("⚠ Usage: /capital <montant>")

    def grid_report(self, update, context):
        if self.bot_logic.grid_lower:
            msg = (
                f"📊 Rapport Grille\n"
                f"Symbole: {self.bot_logic.current_symbol}\n"
                f"Plage: {self.bot_logic.grid_lower:.2f} - {self.bot_logic.grid_upper:.2f}\n"
                f"Niveaux: {self.bot_logic.grid_levels}\n"
                f"Taille ordre: {self.bot_logic.order_amount:.6f}"
            )
        else:
            msg = "⚠ Grille non initialisée"
        self.send_message(msg)

    def start(self):
        self.updater.start_polling()

# --- Configuration finale ---
if __name__ == "__main__":
    api_key = os.getenv("HYPERLIQUID_API_KEY")
    api_secret = os.getenv("HYPERLIQUID_SECRET")
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    exchange = HyperliquidAPI(api_key, api_secret)
    bot = GridTradingBot(
        exchange=exchange,
        capital=1000,
        grid_levels=10,
        dry_run=False,
        scan_interval=3600
    )

    telegram = TelegramInterface(telegram_token, telegram_chat_id, bot)
    bot.telegram_bot = telegram

    threading.Thread(target=telegram.start).start()
    bot.run()
