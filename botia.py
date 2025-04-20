import logging
import os
import numpy as np
import pandas as pd
from functools import lru_cache
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler
)
from pycoingecko import CoinGeckoAPI
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import json
from openai import OpenAI

# Configuration
ASK_TOKEN = 0
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-1")
MODELS_DIR = 'models'
HISTORY_FILE = 'analysis_history.json'
os.makedirs(MODELS_DIR, exist_ok=True)

cg = CoinGeckoAPI()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Helper functions
def convert_to_float(value):
    if isinstance(value, np.float32):
        return float(value)
    return value

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([convert_to_float(item) for item in history], f, indent=4)

# Module Grok
class GrokClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=GROK_API_KEY,
            base_url="https://api.x.ai/v1"
        )
    
    def generate_response(self, user_input: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=GROK_MODEL,
                messages=[
                    {"role": "system", "content": "Expert en trading crypto. Sois concis et technique."},
                    {"role": "user", "content": user_input}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Erreur Grok: {str(e)}")
            return "❌ Erreur de réponse Grok"

# Fonctions trading
@lru_cache(maxsize=100)
def get_crypto_data(token: str, days: int):
    try:
        return cg.get_coin_ohlc_by_id(id=token, vs_currency='usd', days=days)
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        return None

def get_live_price(token: str):
    try:
        return cg.get_price(ids=token, vs_currencies='usd')[token]['usd']
    except:
        return None

def compute_macd(data):
    return ta.macd(data['close'])

def compute_rsi(data):
    return ta.rsi(data['close'])

def generate_labels(df):
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['label'] = np.where(df['return'] > 0.003, 2, np.where(df['return'] < -0.003, 0, 1))
    return df.dropna()

# Handlers Telegram
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("👋 Quel token veux-tu analyser ? (ex: bitcoin)")
    return ASK_TOKEN

async def ask_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    token = update.message.text.strip().lower()
    await analyze_and_reply(update, token)
    return ConversationHandler.END

async def analyze_and_reply(update: Update, token: str):
    await update.message.reply_text(f"📈 Analyse de {token} en cours...")
    try:
        ohlc = get_crypto_data(token, 30)
        if not ohlc:
            await update.message.reply_text("❌ Token non trouvé")
            return

        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = generate_labels(df)

        # ... (ajouter le reste de la logique d'analyse)

        live_price = get_live_price(token)
        message = f"💰 Prix live: {live_price:.2f}$" if live_price else "⚠️ Prix non disponible"
        await update.message.reply_text(message)

    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {str(e)}")

async def grok_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    grok = GrokClient()
    response = grok.generate_response(update.message.text)
    await update.message.reply_text(response[:4000])

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    history = load_history()
    if history:
        await update.message.reply_text("\n\n".join(
            f"{item['token']} - {item['timestamp']}" for item in history[-5:]
        ))
    else:
        await update.message.reply_text("Aucun historique")

def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={ASK_TOKEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_token)]},
        fallbacks=[CommandHandler('history', show_history)]
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('history', show_history))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, grok_handler))
    
    application.run_polling()

if __name__ == '__main__':
    main()
