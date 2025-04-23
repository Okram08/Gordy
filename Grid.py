import os
import pandas as pd
import numpy as np
import time
import logging
import threading
from datetime import datetime
from telegram import Bot
from telegram.ext import Updater, CommandHandler
from dotenv import load_dotenv
import ta
import ccxt

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

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=500):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    def fetch_open_orders(self, symbol):
        return self.exchange.fetch_open_orders(symbol)

    def cancel_order(self, order_id, symbol):
        return self.exchange.cancel_order(order_id, symbol)

    def create_limit_buy_order(self, symbol, amount, price):
        return self.exchange.create_order(symbol, 'limit', 'buy', amount, price, params={'timeInForce': 'PostOnly'})

    def create_limit_sell_order(self, symbol, amount, price):
        return self.exchange.create_order(symbol, 'limit', 'sell', amount, price, params={'timeInForce': 'PostOnly'})

    def fetch_closed_orders(self, symbol, since):
        return self.exchange.fetch_closed_orders(symbol, since=since)

    def milliseconds(self):
        return self.exchange.milliseconds()

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
            f"\U0001F4CA Status Bot\n"
            f"Actif: {self.bot_logic.current_symbol or 'Aucun'}\n"
            f"Capital: {self.bot_logic.capital:.2f} USDT\n"
            f"Performance: +{(self.bot_logic.total_sell_revenue - self.bot_logic.total_buy_cost - self.bot_logic.total_fees):.2f} USDT\n"
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
        self.send_message("\U0001F6D1 Arrêt complet du bot...")
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
            self.send_message(f"\U0001F4B0 Capital mis à jour: {new_capital} USDT")
        except:
            self.send_message("⚠ Usage: /capital <montant>")

    def grid_report(self, update, context):
        if self.bot_logic.grid_lower:
            msg = (
                f"\U0001F4CA Rapport Grille\n"
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
        symbols=['BTC/USDT:USDT', 'ETH/USDT:USDT'],
        capital=1000,
        grid_levels=10,
        dry_run=False,
        scan_interval=3600
    )

    telegram = TelegramInterface(telegram_token, telegram_chat_id, bot)
    bot.telegram_bot = telegram

    threading.Thread(target=telegram.start).start()
    bot.run()
