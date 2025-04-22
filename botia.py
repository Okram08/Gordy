import logging
import os
import requests
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

ASK_TOKEN = 0
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

cg = CoinGeckoAPI()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

LOOKBACK = 24
TRAIN_TEST_RATIO = 0.8
CLASS_THRESHOLD = 0.003
HISTORY_FILE = 'analysis_history.json'


# Fonction pour charger l'historique des analyses
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
                logging.info(f"Historique chargé avec {len(history)} éléments.")
                return history
        except json.JSONDecodeError:
            logging.error(f"Erreur de formatage dans le fichier {HISTORY_FILE}, réinitialisation.")
            with open(HISTORY_FILE, 'w') as f:
                json.dump([], f)
            return []
    else:
        logging.info(f"Aucun fichier historique trouvé, création de {HISTORY_FILE}.")
        return []


# Fonction pour récupérer les données des crypto-monnaies
def get_crypto_data(token: str, days: int):
    try:
        if days > 90:
            days = 90
        return cg.get_coin_ohlc_by_id(id=token, vs_currency='usd', days=days)
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des données pour {token}: {str(e)}")
        return None


# Fonction pour obtenir le prix actuel d'une crypto-monnaie
def get_live_price(token: str):
    try:
        data = cg.get_price(ids=token, vs_currencies='usd')
        return data[token]['usd'] if token in data else None
    except Exception as e:
        logging.error(f"Erreur API prix live pour {token}: {str(e)}")
        return None


# Fonction pour calculer le MACD
def compute_macd(data):
    short_ema = data.ewm(span=12, adjust=False).mean()
    long_ema = data.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


# Fonction pour générer les labels (buy, sell, neutral)
def generate_labels(df):
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['label'] = 1 * (df['return'] > CLASS_THRESHOLD) + (-1) * (df['return'] < -CLASS_THRESHOLD)
    df.dropna(inplace=True)
    df['label'] = df['label'] + 1
    return df


# Fonction pour préparer les données d'entraînement et de test
def prepare_data(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(LOOKBACK, len(df_scaled)):
        X.append(df_scaled[i - LOOKBACK:i])
        y.append(df['label'].values[i])

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=3)
    return train_test_split(X, y, test_size=1 - TRAIN_TEST_RATIO, shuffle=False)


# Fonction de gestion des réponses Rasa et de récupération des demandes crypto
def get_rasa_response(message):
    try:
        url = "http://localhost:5005/webhooks/rest/webhook"  # URL de ton serveur Rasa
        payload = {
            "sender": "telegram_user",
            "message": message
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            rasa_messages = response.json()
            if rasa_messages:
                return rasa_messages[0].get('text', '').lower()
            return None
        else:
            logging.error(f"Erreur avec Rasa API: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Erreur lors de l'appel à Rasa: {str(e)}")
        return None


# Fonction principale d'analyse de crypto-monnaie
async def analyze_and_reply(update: Update, token: str):
    rasa_message = get_rasa_response(token)

    if rasa_message:
        # Si la réponse est liée à l'analyse de crypto-monnaie, on passe à l'analyse
        if "analyser" in rasa_message.lower():
            await update.message.reply_text(f"📈 {rasa_message}")
            await perform_crypto_analysis(update, token)
        else:
            # Si Rasa a renvoyé autre chose (par exemple, une réponse générique), on envoie la réponse de Rasa
            await update.message.reply_text(f"Rasa dit : {rasa_message}")
    else:
        # Sinon, effectuer l'analyse comme précédemment
        await perform_crypto_analysis(update, token)


# Fonction pour effectuer l'analyse de la crypto-monnaie
async def perform_crypto_analysis(update: Update, token: str):
    await update.message.reply_text(f"📈 Analyse de {token} en cours...")

    try:
        ohlc = get_crypto_data(token, 30)
        if not ohlc:
            await update.message.reply_text("❌ Token non trouvé ou erreur API.")
            return

        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        df['macd'], df['signal'] = compute_macd(df['close'])
        df = generate_labels(df)

        current_price = get_live_price(token)
        if current_price is None:
            await update.message.reply_text("❌ Impossible de récupérer le prix en direct. Réessaie plus tard.")
            return

        direction = "⬆️ LONG" if df['label'].iloc[-1] == 2 else ("⬇️ SHORT" if df['label'].iloc[-1] == 0 else "🔁 NEUTRE")
        message = (
            f"📊 {token.upper()} - Signal IA\n"
            f"🎯 Direction: {direction}\n"
            f"💰 Prix live: {current_price:.2f}$\n"
        )

        history = load_history()
        result = {
            'token': token,
            'timestamp': str(datetime.now()),
            'direction': direction,
            'current_price': float(current_price),
        }
        history.append(result)
        save_history(history)

        await update.message.reply_text(message)

    except Exception as e:
        logging.error(f"Erreur: {str(e)}")
        await update.message.reply_text(f"❌ Une erreur est survenue durant l'analyse.\n🛠 Détail: {str(e)}")


# Fonction pour commencer une conversation
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("👋 Quel token veux-tu analyser (ex: bitcoin) ?")
    return ASK_TOKEN


# Fonction pour gérer la demande de token
async def ask_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    token = update.message.text.strip().lower()
    await analyze_and_reply(update, token)
    return ConversationHandler.END


# Fonction pour afficher l'historique des analyses
async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    history = load_history()
    if history:
        messages = [
            f"🕒 {entry['timestamp']}\n📉 {entry['token'].upper()} | {entry['direction']}\n"
            for entry in history[-5:]
        ]
        await update.message.reply_text("\n\n".join(messages))
    else:
        await update.message.reply_text("Aucune analyse historique disponible.")


# Fonction principale de configuration du bot Telegram
def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("history", show_history))
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={ASK_TOKEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_token)]},
        fallbacks=[]
    )
    application.add_handler(conv_handler)
    application.run_polling()


if __name__ == '__main__':
    main()
