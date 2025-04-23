import os
import time
import logging
import threading
import ccxt
import pandas as pd
import numpy as np
import ta
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
        self.load_markets()

    def load_markets(self):
        self.exchange.load_markets()
        logging.info("Marchés chargés avec succès")

    def get_perpetual_symbols(self):
        return [symbol for symbol in self.exchange.symbols if 'USD:USD' in symbol]

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # Autres méthodes d'API...

class GridTradingBot:
    def __init__(self, api, capital=1000, grid_levels=10, scan_interval=3600):
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
            
            # Calcul des indicateurs
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
            df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
            
            last = df.iloc[-1]
            
            # Score de trading (plus bas = mieux)
            score = 0
            score += abs(last['rsi'] - 50) * 0.4  # Neutralité RSI
            score += (last['bb_upper'] - last['bb_lower']) / last['close'] * 100 * 0.3  # Volatilité
            score += (30 - min(last['adx'], 30)) * 0.3  # Faible tendance
            
            return score
            
        except Exception as e:
            logging.error(f"Erreur analyse {symbol}: {e}")
            return float('inf')

    def select_best_symbol(self):
        symbols = self.api.get_perpetual_symbols()
        scores = {}
        
        for symbol in symbols:
            score = self.analyze_symbol(symbol)
            scores[symbol] = score
            logging.info(f"Score {symbol}: {score:.2f}")
            time.sleep(0.5)  # Respect rate limits
            
        return min(scores, key=scores.get) if scores else None

    def calculate_grid(self, symbol):
        ticker = self.api.exchange.fetch_ticker(symbol)
        price = ticker['last']
        volatility = (ticker['high'] - ticker['low']) / ticker['low']
        
        # Ajustement dynamique de la grille
        self.grid_lower = price * (1 - volatility)
        self.grid_upper = price * (1 + volatility)
        self.grid_size = (self.grid_upper - self.grid_lower) / self.grid_levels
        self.order_amount = (self.capital * 0.95) / (self.grid_levels * price)

    def run_strategy(self):
        while self.running:
            try:
                best_symbol = self.select_best_symbol()
                
                if best_symbol != self.current_symbol:
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

    # Méthodes restantes (place_grid_orders, gestion risques, etc.)...

class TelegramInterface:
    # Implementation similaire à précédemment...

if __name__ == "__main__":
    api = HyperliquidAPI(
        api_key=os.getenv("HYPERLIQUID_API_KEY"),
        api_secret=os.getenv("HYPERLIQUID_SECRET")
    )
    
    bot = GridTradingBot(api, capital=50)
    telegram = TelegramInterface(os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), bot)
    bot.telegram_bot = telegram
    
    threading.Thread(target=telegram.start).start()
    bot.run_strategy()
