import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import httpx
import json

# === CONFIG ===
TOKEN = os.getenv('TELEGRAM_TOKEN') or 'YOUR_TELEGRAM_BOT_TOKEN'
MODELS_DIR = './models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

logging.basicConfig(level=logging.INFO)

# === TELEGRAM COMMAND ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Salut ! Envoie-moi un nom de crypto comme /analyse BTC pour commencer.")

async def analyse(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❗ Usage: /analyse BTC")
        return
    token = context.args[0].lower()
    await analyze_and_reply(update, token)

# === API COINGECKO ===
def get_crypto_data(symbol: str, days: int):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/ohlc?vs_currency=usd&days={min(days,90)}"
    try:
        response = httpx.get(url)
        data = response.json()
        return data
    except Exception as e:
        logging.error(f"API Error for {symbol}: {e}")
        return None

def get_live_price(symbol: str):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
    try:
        response = httpx.get(url)
        return response.json()[symbol]['usd']
    except:
        return None

# === TECHNICAL INDICATORS ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# === LABEL GENERATION ===
def generate_labels(df):
    df['future'] = df['close'].shift(-1)
    df['label'] = 1
    df.loc[df['future'] > df['close'] * 1.005, 'label'] = 2
    df.loc[df['future'] < df['close'] * 0.995, 'label'] = 0
    return df.dropna()

# === DATA PREPARATION ===
def prepare_data(df, features):
    df = df.dropna()
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    X = []
    y = []
    window = 10
    for i in range(len(df) - window):
        X.append(df[features].iloc[i:i+window].values)
        y.append(df['label'].iloc[i+window])
    X = np.array(X)
    y = np.array(y)
    y = np.eye(3)[y]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# === HISTORY ===
def load_history():
    if os.path.exists('history.json'):
        with open('history.json') as f:
            return json.load(f)
    return []

def save_history(data):
    with open('history.json', 'w') as f:
        json.dump(data, f, indent=4)

# === AI ANALYSIS FUNCTION ===
async def analyze_and_reply(update: Update, token: str):
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

        if df.empty or 'label' not in df.columns or df['label'].nunique() < 2:
            await update.message.reply_text("❌ Pas assez de données ou de signaux pour entraîner le modèle.")
            return

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

# === MAIN ===
if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyse", analyse))
    print("✅ Bot prêt")
    app.run_polling()
