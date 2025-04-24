import logging
import os
import re
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
    ConversationHandler,
    ContextTypes
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

# États de la conversation
ASK_TOKEN, ASK_CONTINUE = range(2)

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
                return json.load(f)
        except json.JSONDecodeError:
            with open(HISTORY_FILE, 'w') as f:
                json.dump([], f)
            return []
    else:
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)
        return []

def save_history(history):
    history = convert_to_float(history)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

@lru_cache(maxsize=100)
def get_crypto_data(token: str, days: int):
    if days > 90:
        days = 90
    return cg.get_coin_ohlc_by_id(id=token, vs_currency='usd', days=days)

def get_live_price(token: str):
    data = cg.get_price(ids=token, vs_currencies='usd')
    return data.get(token, {}).get('usd')

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
    df['label'] += 1
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
    await update.message.reply_text(
        "👋 Bonjour ! Indique-moi les tokens à analyser (séparés par des virgules)."
    )
    return ASK_TOKEN

async def ask_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip().lower()
    tokens = re.split(r'[\s,;]+', text)
    for token in tokens:
        await analyze_and_reply(update, token)
    await update.message.reply_text("✅ Analyses terminées.\nVoulez-vous analyser d'autres tokens ? (Oui / Non)")
    return ASK_CONTINUE

async def analyze_and_reply(update: Update, token: str):
    await update.message.reply_text(f"📈 Analyse de {token} en cours…")
    try:
        ohlc = get_crypto_data(token, 30)
        if not ohlc:
            await update.message.reply_text(f"❌ Token « {token} » non trouvé.")
            return

        df = pd.DataFrame(ohlc, columns=['timestamp','open','high','low','close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        df['macd'], df['signal'] = compute_macd(df['close'])
        df['rsi'] = compute_rsi(df['close'])
        df['atr'] = compute_atr(df['high'], df['low'], df['close'])
        df = generate_labels(df)

        X_train, X_test, y_train, y_test = prepare_data(df, ['rsi','macd','atr'])
        model_path = os.path.join(MODELS_DIR, f"{token}_clf_model.keras")
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([
                Input(shape=(X_train.shape[1], X_train.shape[2])),
                LSTM(64, return_sequences=True), Dropout(0.3),
                LSTM(32), Dropout(0.2),
                Dense(3, activation='softmax')
            ])
            model.compile('adam','categorical_crossentropy',['accuracy'])
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            model.save(model_path)

        pred = model.predict(X_test[-1:], verbose=0)[0]
        cls = np.argmax(pred)
        confidence = pred[cls]
        direction = "⬆️ LONG" if cls==2 else "⬇️ SHORT" if cls==0 else "🔁 NEUTRE"

        price = get_live_price(token)
        if price is None:
            await update.message.reply_text("❌ Impossible de récupérer le prix live.")
            return
        atr = df['atr'].iloc[-1]
        tp = price + 2*atr if cls==2 else price - 2*atr if cls==0 else price
        sl = price - atr if cls==2 else price + atr if cls==0 else price

        message = (
            f"📊 {token.upper()} — Signal IA\n"
            f"🎯 Direction : {direction}\n"
            f"📈 Confiance : {confidence*100:.2f}%\n"
            f"💰 Prix live : {price:.2f}$\n"
            f"🎯 TP : {tp:.2f}$ | 🛑 SL : {sl:.2f}$"
        )
        await update.message.reply_text(message)

        h = load_history()
        h.append({
            'token': token,
            'timestamp': datetime.now().isoformat(),
            'direction': direction,
            'confidence': confidence,
            'current_price': price,
            'tp': tp,
            'sl': sl
        })
        save_history(h)

    except Exception as e:
        logging.error(f"Erreur analyse {token} : {e}")
        await update.message.reply_text(f"❌ Erreur durant l'analyse de {token}.")

async def continue_predictions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    answer = update.message.text.strip().lower()
    if answer in ('oui', 'yes'):
        await update.message.reply_text("🔄 D'accord, quels tokens cette fois ?")
        return ASK_TOKEN
    else:
        await update.message.reply_text("👍 Très bien, à bientôt !")
        return ConversationHandler.END

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    h = load_history()
    if not h:
        await update.message.reply_text("Aucune analyse historique disponible.")
        return
    lines = []
    for e in h[-5:]:
        lines.append(
            f"🕒 {e['timestamp']}\n"
            f"{e['token'].upper()} | {e['direction']} | {e['confidence']*100:.2f}%\n"
            f"Prix : {e['current_price']:.2f}$ | TP : {e['tp']:.2f}$ | SL : {e['sl']:.2f}$"
        )
    await update.message.reply_text("\n\n".join(lines))

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("history", show_history))
    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_TOKEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_token)],
            ASK_CONTINUE: [MessageHandler(filters.Regex(re.compile(r'^(oui|non|yes|no)$', re.IGNORECASE)), continue_predictions)]
        },
        fallbacks=[CommandHandler("start", start)]
    )
    app.add_handler(conv)
    app.run_polling()

if __name__ == "__main__":
    main()
