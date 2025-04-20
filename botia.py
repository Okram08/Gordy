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
from transformers import pipeline

# Constants
ASK_TOKEN = 0
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialiser CoinGeckoAPI et Hugging Face (GPT-2)
cg = CoinGeckoAPI()
generator = pipeline('text-generation', model='gpt2')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

LOOKBACK = 24
TRAIN_TEST_RATIO = 0.8
CLASS_THRESHOLD = 0.003
HISTORY_FILE = 'analysis_history.json'


def convert_to_float(value):
    if isinstance(value, (np.float32, np.float64, np.int64)):
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


def save_history(history):
    history = convert_to_float(history)
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
        logging.info(f"Historique sauvegardé avec {len(history)} éléments.")
    except Exception as e:
        logging.error(f"Erreur lors de l'écriture dans le fichier JSON : {str(e)}")


@lru_cache(maxsize=100)
def get_crypto_data(token: str, days: int):
    try:
        return cg.get_coin_ohlc_by_id(id=token, vs_currency='usd', days=days)
    except Exception as e:
        logging.error(f"API Error for {token}: {str(e)}")
        return None


def get_live_price(token: str):
    try:
        data = cg.get_price(ids=token, vs_currencies='usd')
        return data[token]['usd'] if token in data else None
    except Exception as e:
        logging.error(f"Erreur API prix live pour {token}: {str(e)}")
        return None


def compute_macd(data):
    short_ema = data.ewm(span=12, adjust=False).mean()
    long_ema = data.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def compute_rsi(data, period=14):
    return ta.rsi(data, length=period)


def compute_atr(high, low, close):
    return ta.atr(high, low, close, length=14)


def generate_labels(df):
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['label'] = 1 * (df['return'] > CLASS_THRESHOLD) + (-1) * (df['return'] < -CLASS_THRESHOLD)
    df.dropna(inplace=True)
    df['label'] = df['label'] + 1
    return df


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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("👋 Quel token veux-tu analyser (ex: bitcoin) ?")
    return ASK_TOKEN


async def ask_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    token = update.message.text.strip().lower()
    await analyze_and_reply(update, token)
    return ConversationHandler.END


async def analyze_and_reply(update: Update, token: str):
    # Vérifier si l'utilisateur demande une analyse via un langage naturel
    if 'analyser' in token or 'analyse' in token:
        await update.message.reply_text(f"📈 Analyse de {token} en cours...")

        try:
            ohlc = get_crypto_data(token, 30)
            if not ohlc:
                await update.message.reply_text("❌ Token non trouvé")
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
                model = Sequential([
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
                await update.message.reply_text("❌ Impossible de récupérer le prix en direct. Réessaie plus tard.")
                return

            atr = df['atr'].iloc[-1]

            tp = current_price + 2 * atr if pred_class == 2 else (current_price - 2 * atr if pred_class == 0 else current_price)
            sl = current_price - atr if pred_class == 2 else (current_price + atr if pred_class == 0 else current_price)

            message = (
                f"📊 {token.upper()} - Signal IA\n"
                f"🎯 Direction: {direction}\n"
                f"📈 Confiance: {confidence*100:.2f}%\n"
                f"💰 Prix live: {current_price:.2f}$\n"
                f"🎯 TP: {tp:.2f}$ | 🛑 SL: {sl:.2f}$\n"
            )

            history = load_history()
            result = {
                'token': token,
                'timestamp': str(datetime.now()),
                'direction': direction,
                'confidence': confidence,
                'pred_class': pred_class,
                'current_price': current_price,
                'tp': tp,
                'sl': sl
            }
            history.append(result)
            save_history(history)

            await update.message.reply_text(message)

        except Exception as e:
            logging.error(f"Erreur: {str(e)}")
            await update.message.reply_text(f"❌ Une erreur est survenue durant l'analyse.\n🛠 Détail: {str(e)}")
    else:
        # Utiliser le modèle GPT-2 pour répondre à des questions générales
        response = generator(token, max_length=100, num_return_sequences=1)[0]['generated_text']
        await update.message.reply_text(response)


async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    history = load_history()
    if history:
        messages = [f"Analyse pour {entry['token']} à {entry['timestamp']}\n"
                    f"Direction: {entry['direction']} | Confiance: {entry['confidence']*100:.2f}%\n"
                    f"Prix actuel: {entry['current_price']}$ | TP: {entry['tp']}$ | SL: {entry['sl']}$\n"
                    for entry in history]
        await update.message.reply_text("\n\n".join(messages))
    else:
        await update.message.reply_text("Aucune analyse historique disponible.")


def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Ajout d'un gestionnaire pour la commande /history
    application.add_handler(CommandHandler("history", show_history))

    # Ajout du ConversationHandler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            ASK_TOKEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_token)],
        },
        fallbacks=[CommandHandler('history', show_history)]
    )

    application.add_handler(conv_handler)

    application.run_polling()


if __name__ == '__main__':
    main()
