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
import time
import matplotlib.pyplot as plt

# Variables et paramètres de configuration
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

# Charger l'historique des analyses
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

# Sauvegarder l'historique des analyses
def save_history(history):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
        logging.info(f"Historique sauvegardé avec {len(history)} éléments.")
    except Exception as e:
        logging.error(f"Erreur lors de l'écriture dans le fichier JSON : {str(e)}")

# Convertir en float
def convert_to_float(value):
    if isinstance(value, (np.float32, np.float64, np.int64)):
        return float(value)
    elif isinstance(value, dict):
        return {k: convert_to_float(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_float(v) for v in value]
    else:
        return value

# Récupérer les données crypto
@lru_cache(maxsize=100)
def get_crypto_data(token: str, days: int):
    try:
        if days > 90:
            days = 90
        return cg.get_coin_ohlc_by_id(id=token, vs_currency='usd', days=days)
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des données pour {token}: {str(e)}")
        return None

# Fonction pour générer un graphique
def generate_price_chart(df, token):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['close'], label='Prix')
    plt.title(f"Évolution du prix de {token.upper()}")
    plt.xlabel('Date')
    plt.ylabel('Prix (USD)')
    plt.legend()
    chart_file = 'price_chart.png'
    plt.savefig(chart_file)
    plt.close()
    return chart_file

# Fonction pour démarrer la conversation
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("👋 Quel(s) token(s) veux-tu analyser ? (ex: bitcoin, ethereum, dogecoin) 📉")
    return ASK_TOKEN

# Fonction pour demander le token à analyser
async def ask_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    tokens = [token.strip().lower() for token in update.message.text.split(',')]
    await update.message.reply_text(f"🔍 Analyse en cours pour les tokens : {', '.join(tokens)}...")
    for token in tokens:
        await analyze_and_reply(update, token)
        time.sleep(2)  # Pause entre les analyses
    return ConversationHandler.END

# Fonction pour analyser un token et renvoyer les résultats
async def analyze_and_reply(update: Update, token: str):
    await update.message.reply_text(f"🔄 Recherche des données pour {token}...")
    try:
        ohlc = get_crypto_data(token, 30)
        if not ohlc:
            await update.message.reply_text(f"❌ Impossible de récupérer les données pour {token}. Vérifiez l'orthographe ou réessayez plus tard.")
            return

        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        df['macd'], df['signal'] = compute_macd(df['close'])
        df['rsi'] = compute_rsi(df['close'])
        df['atr'] = compute_atr(df['high'], df['low'], df['close'])
        df = generate_labels(df)

        features = ['rsi', 'macd', 'atr']
        X_train, X_test, y_train, y_test = prepare_data(df, features)

        model_path = os.path.join(MODELS_DIR, f'{token}_clf_model.keras')

        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([  # Création et entraînement du modèle
                Input(shape=(X_train.shape[1], X_train.shape[2])),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.2),
                Dense(3, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            model.save(model_path)

        last_sequence = X_test[-1:]
        prediction = model.predict(last_sequence, verbose=0)[0]
        pred_class = np.argmax(prediction)
        confidence = prediction[pred_class]

        direction = "⬆️ LONG" if pred_class == 2 else ("⬇️ SHORT" if pred_class == 0 else "🔁 NEUTRE")

        current_price = get_live_price(token)
        if current_price is None:
            await update.message.reply_text(f"❌ Impossible de récupérer le prix en direct pour {token}. Réessaie plus tard.")
            return

        atr = df['atr'].iloc[-1]
        tp = current_price + 2 * atr if pred_class == 2 else (current_price - 2 * atr if pred_class == 0 else current_price)
        sl = current_price - atr if pred_class == 2 else (current_price + atr if pred_class == 0 else current_price)

        message = (
            f"📊 {token.upper()} - Signal IA\n"
            f"🎯 Direction: {direction}\n"
            f"📈 Confiance: {confidence * 100:.2f}%\n"
            f"💰 Prix live: {current_price:.2f}$\n"
            f"🎯 TP: {tp:.2f}$ | 🛑 SL: {sl:.2f}$\n"
            f"\n📊 Graphique disponible sur demande !"
        )

        history = load_history()
        result = {
            'token': token,
            'timestamp': str(datetime.now()),
            'direction': direction,
            'confidence': confidence,
            'pred_class': int(pred_class),
            'current_price': float(current_price),
            'tp': float(tp),
            'sl': float(sl)
        }
        history.append(result)
        save_history(history)

        # Envoi du graphique
        chart_file = generate_price_chart(df, token)
        await update.message.reply_photo(photo=open(chart_file, 'rb'))

        await update.message.reply_text(message)

    except Exception as e:
        logging.error(f"Erreur: {str(e)}")
        await update.message.reply_text(f"❌ Une erreur est survenue durant l'analyse.\n🛠 Détail: {str(e)}")
