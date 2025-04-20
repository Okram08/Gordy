import logging
import os
import numpy as np
import pandas as pd
from io import BytesIO
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

# Configuration initiale
ASK_TOKEN = 0
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-1")
MODELS_DIR = 'models'
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
    elif isinstance(value, dict):
        return {k: convert_to_float(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_float(v) for v in value]
    else:
        return value

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            with open(HISTORY_FILE, 'w') as f:
                json.dump([], f)
            return []
    else:
        return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(convert_to_float(history), f, indent=4)

# Module Grok AI
class GrokClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=GROK_API_KEY,
            base_url="https://api.x.ai/v1"
        )
    
    def generate_response(self, user_input: str) -> str:
        history = load_history()[-3:]  # Derniers 3 trades
        context = {
            "history": history,
            "btc_price": get_live_price('bitcoin'),
            "eth_price": get_live_price('ethereum')
        }
        
        system_prompt = f"""Tu es un expert en trading crypto. Analyse les données suivantes :
        Historique récent : {context['history']}
        Données live : BTC={context['btc_price']}$, ETH={context['eth_price']}$
        Réponds en français de manière concise et technique."""
        
        try:
            response = self.client.chat.completions.create(
                model=GROK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Erreur Grok: {str(e)}")
            return "❌ Désolé, je ne peux pas répondre pour le moment."

# Core trading functions
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

# ... (Conserver toutes les fonctions existantes : compute_macd, compute_rsi, generate_labels, etc.)

# Handlers Telegram
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("👋 Quel token veux-tu analyser ? (ex: bitcoin)")
    return ASK_TOKEN

async def analyze_and_reply(update: Update, token: str):
    # ... (Conserver la logique existante de analyse_and_reply)
    # Modifier uniquement la partie prix :
    live_price = get_live_price(token)
    current_price = live_price if live_price else df['close'].iloc[-1]
    # ... (suite du code existant)

async def grok_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    grok = GrokClient()
    
    if user_input.lower().startswith('/ask'):
        query = user_input[4:].strip()
        response = grok.generate_response(query)
    else:
        response = grok.generate_response(user_input)
    
    await update.message.reply_text(response[:4000])  # Limite Telegram

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    history = load_history()
    # ... (Conserver la logique existante)

def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Conversation pour l'analyse
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={ASK_TOKEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_token)]},
        fallbacks=[CommandHandler('history', show_history)]
    )

    # Handler Grok pour les messages libres
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & ~filters.Regex(r'^/'),
        grok_handler
    ))

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('history', show_history))
    application.run_polling()

if __name__ == '__main__':
    main()
