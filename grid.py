import os
import time
import logging
import threading
import ccxt
import pandas as pd
import ta
from telegram import Bot
from telegram.ext import Updater, CommandHandler
from dotenv import load_dotenv

# Config de base
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HyperliquidAPI:
    def __init__(self, api_key, api_secret):
        self.exchange = ccxt.hyperliquid({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}  # ✅ SPOT ici
        })
        self.load_markets()

    def load_markets(self):
        self.exchange.load_markets()
        logging.info("Marchés chargés avec succès")

    def get_spot_symbols(self):
        return [s for s in self.exchange.symbols if s.endswith('/USDT')]

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


class GridTradingBot:
    def __init__(self, api, capital=1000, grid_levels=10, scan_interval=60):
        self.api = api
        self.capital = capital
        self.grid_levels = grid_levels
        self.scan_interval = scan_interval
        self.running = True
        self.current_symbol = None
        self.telegram_bot = None

    def analyze_symbol(self, symbol):
        try:
            ohlcv = self.api.fetch_ohlcv(symbol)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

            last = df.iloc[-1]

            score = 0
            score += abs(last['rsi'] - 50) * 0.4
            score += (last['bb_upper'] - last['bb_lower']) / last['close'] * 100 * 0.3
            score += (30 - min(last['adx'], 30)) * 0.3

            return score

        except Exception as e:
            logging.error(f"Erreur analyse {symbol}: {e}")
            return float('inf')

    def select_best_symbol(self):
        symbols = self.api.get_spot_symbols()
        scores = {}

        for symbol in symbols:
            try:
                ticker = self.api.exchange.fetch_ticker(symbol)
                volume = ticker.get('quoteVolume', 0)

                if volume < 1_000_000:
                    logging.info(f"⛔ {symbol} ignoré (volume trop faible: {volume:.0f})")
                    continue

                score = self.analyze_symbol(symbol)
                scores[symbol] = score
                logging.info(f"✅ {symbol} | Volume: {volume:.0f} | Score: {score:.2f}")
                time.sleep(0.5)

            except Exception as e:
                logging.warning(f"⚠️ Erreur ticker {symbol}: {e}")

        return min(scores, key=scores.get) if scores else None

    def calculate_grid(self, symbol):
        ticker = self.api.exchange.fetch_ticker(symbol)
        price = ticker['last']
        volatility = (ticker['high'] - ticker['low']) / ticker['low']

        self.grid_lower = price * (1 - volatility)
        self.grid_upper = price * (1 + volatility)
        self.grid_size = (self.grid_upper - self.grid_lower) / self.grid_levels
        self.order_amount = (self.capital * 0.95) / (self.grid_levels * price)

    def place_grid_orders(self):
        logging.info("📌 Placement des ordres fictifs (simulation)")
        for i in range(self.grid_levels + 1):
            price = self.grid_lower + i * self.grid_size
            logging.info(f"💸 Ordre fictif à {price:.4f} pour {self.order_amount:.4f} unités")

    def run_strategy(self):
        logging.info("Scheduler started")
        while self.running:
            try:
                logging.info("🔍 Sélection du meilleur symbole...")
                best_symbol = self.select_best_symbol()
                logging.info(f"✅ Meilleur symbole : {best_symbol}")

                if best_symbol and best_symbol != self.current_symbol:
                    self.current_symbol = best_symbol
                    self.calculate_grid(best_symbol)
                    self.place_grid_orders()

                    if self.telegram_bot:
                        self.telegram_bot.send_message(
                            f"🔄 Nouveau symbole sélectionné: {best_symbol}\n"
                            f"📊 Fourchette: {self.grid_lower:.2f} - {self.grid_upper:.2f}"
                        )

                time.sleep(self.scan_interval)

            except Exception as e:
                logging.error(f"Erreur stratégie: {e}")
                time.sleep(60)


class TelegramInterface:
    def __init__(self, token, chat_id, bot_logic):
        self.token = token
        self.chat_id = chat_id
        self.bot_logic = bot_logic
        self.bot = Bot(token=self.token)
        self.updater = Updater(token=self.token, use_context=True)
        dp = self.updater.dispatcher

        dp.add_handler(CommandHandler("start", self.start_command))

    def start_command(self, update, context):
        actual_chat_id = update.effective_chat.id
        context.bot.send_message(chat_id=actual_chat_id, text=f"🤖 Bot actif !\nTon chat_id est : {actual_chat_id}")

    def send_message(self, message):
        self.bot.send_message(chat_id=self.chat_id, text=message)

    def start(self):
        logging.info("📬 Interface Telegram démarrée")
        self.updater.start_polling()


if __name__ == "__main__":
    api = HyperliquidAPI(
        api_key=os.getenv("HYPERLIQUID_API_KEY"),
        api_secret=os.getenv("HYPERLIQUID_SECRET")
    )

    bot = GridTradingBot(api, capital=100, scan_interval=60)
    telegram = TelegramInterface(os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), bot)
    bot.telegram_bot = telegram

    threading.Thread(target=telegram.start).start()
    bot.run_strategy()
