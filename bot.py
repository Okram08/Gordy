import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from functools import lru_cache
from dotenv import load_dotenv
from telegram import Update, InputFile
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    filters, 
    ContextTypes, 
    ConversationHandler,
    Defaults
)
from pycoingecko import CoinGeckoAPI
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# === ÉTAT DE CONVERSATION ===
CHOOSING = 0

# Configuration initiale
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

cg = CoinGeckoAPI()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

# Hyperparamètres DAY TRADING
LOOKBACK = 24  # 24 heures (au lieu de jours)
ATR_PERIOD = 14  # 14 périodes (heures)
RISK_PERCENT = 0.01  # 1% de risque par trade
TRAIN_TEST_RATIO = 0.9  # Plus de données récentes pour le test

@lru_cache(maxsize=100)
def get_crypto_data(token: str, days: int = 7):  # Données sur 7 jours
    """Récupère les données horaires"""
    try:
        return cg.get_coin_ohlc_by_id(
            id=token, 
            vs_currency='usd', 
            days=days, 
            interval='hourly'  <-- Données horaires
        )
    except Exception as e:
        logging.error(f"API Error for {token}: {str(e)}")
        return None

def compute_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, period=14):
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1)
    return tr.max(axis=1).rolling(window=period).mean()

async def analyze_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_input = update.message.text.lower()
    await update.message.reply_chat_action(action='typing')

    try:
        # Récupération données HORAIRES
        ohlc = get_crypto_data(user_input, days=7)
        if not ohlc or len(ohlc) < 168:  # 7 jours * 24h
            await update.message.reply_text("❌ Données insuffisantes. Essayez avec une crypto plus liquide.")
            return ConversationHandler.END

        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Features pour day trading
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(4).std()  # Volatilité sur 4h
        df['volume_ma'] = df['close'].rolling(24).mean()  # Volume moyen sur 24h
        df['rsi'] = compute_rsi(df['close'], 14)
        df['atr'] = compute_atr(df['high'], df['low'], df['close'], ATR_PERIOD)
        df.dropna(inplace=True)

        if len(df) < LOOKBACK * 2:
            await update.message.reply_text("❌ Données insuffisantes après nettoyage.")
            return ConversationHandler.END

        # Normalisation
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['close', 'volatility', 'rsi', 'atr']])

        # Séquences pour LSTM
        X, y = [], []
        for i in range(LOOKBACK, len(scaled_data)):
            X.append(scaled_data[i-LOOKBACK:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        # Split train/test
        split = int(TRAIN_TEST_RATIO * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Modèle simplifié
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(32, return_sequences=True),
            Dropout(0.4),  # Réduit l'overfitting
            LSTM(16),
            Dropout(0.3),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Entraînement avec early stopping
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_split=0.1,
            verbose=0
        )

        # Backtesting réaliste
        test_predictions = model.predict(X_test)
        test_accuracy = np.mean(np.sign(test_predictions.flatten()) == np.sign(y_test)) * 100

        # Dernière séquence
        last_sequence = scaled_data[-LOOKBACK:]
        predicted_price = model.predict(np.array([last_sequence]))[0][0]
        predicted_price = scaler.inverse_transform([[predicted_price, 0, 0, 0]])[0][0]

        # Calcul TP/SL en %
        current_price = df['close'].iloc[-1]
        direction = "LONG 🟢" if predicted_price > current_price else "SHORT 🔴"
        tp = current_price * (1 + RISK_PERCENT) if direction == "LONG 🟢" else current_price * (1 - RISK_PERCENT)
        sl = current_price * (1 - RISK_PERCENT) if direction == "LONG 🟢" else current_price * (1 + RISK_PERCENT)

        # Visualisation
        fig, ax = plt.subplots(figsize=(10, 6))
        df['close'].iloc[-48:].plot(ax=ax, title='Prix (48 dernières heures)')  <-- Focus sur court terme
        ax.axhline(tp, color='green', ls='--', label='TP')
        ax.axhline(sl, color='red', ls='--', label='SL')
        ax.legend()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Message
        message = (
            f"📊 {user_input.upper()} - Analyse Day Trading\n\n"
            f"🎯 Direction: {direction}\n"
            f"💰 Prix actuel: {current_price:.2f}$\n"
            f"📈 TP: {tp:.2f}$ ({RISK_PERCENT*100:.1f}%)\n"
            f"📉 SL: {sl:.2f}$ ({RISK_PERCENT*100:.1f}%)\n"
            f"📊 Précision backtest: {test_accuracy:.1f}%\n"
            f"⚡ Volatilité (ATR 14h): {df['atr'].iloc[-1]:.2f}$"
        )

        await update.message.reply_photo(
            photo=InputFile(buf, filename='analysis.png'),
            caption=message
        )
        buf.close()

    except Exception as e:
        logging.exception("Erreur critique:")
        await update.message.reply_text(f"❌ Erreur: {str(e)[:200]}")

    return ConversationHandler.END

# ... (Le reste du code main() reste inchangé) ...
