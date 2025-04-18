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

# Hyperparamètres
LOOKBACK = 48
ATR_PERIOD = 14
RISK_REWARD_RATIO = 2
TRAIN_TEST_RATIO = 0.8

@lru_cache(maxsize=100)
def get_crypto_data(token: str, days: int = 30):
    """Cache les requêtes API avec mémoization"""
    try:
        return cg.get_coin_ohlc_by_id(id=token, vs_currency='usd', days=days)
    except Exception as e:
        logging.error(f"API Error for {token}: {str(e)}")
        return None

# Custom functions to replace talib
def compute_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data, short_period=12, long_period=26, signal_period=9):
    short_ema = data.ewm(span=short_period, min_periods=1, adjust=False).mean()
    long_ema = data.ewm(span=long_period, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_period, min_periods=1, adjust=False).mean()
    return macd, signal_line

def compute_atr(high, low, close, period=14):
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1)
    tr_max = tr.max(axis=1)
    atr = tr_max.rolling(window=period).mean()
    return atr

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "🚀 Crypto Trading Bot Pro\n"
        "Entrez le symbole d'une cryptomonnaie (ex: bitcoin):"
    )
    return CHOOSING

async def analyze_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_input = update.message.text.lower()
    await update.message.reply_chat_action(action='typing')

    try:
        # Récupération des données
        ohlc = get_crypto_data(user_input, days=90)
        if not ohlc:
            raise ValueError("Cryptomonnaie non trouvée")

        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Calcul des indicateurs (RSI, MACD, ATR)
        df['rsi'] = compute_rsi(df['close'], 14)
        df['macd'], df['signal'] = compute_macd(df['close'])
        df['atr'] = compute_atr(df['high'], df['low'], df['close'], ATR_PERIOD)
        df.dropna(inplace=True)

        # Séparation train/test
        train_size = int(len(df) * TRAIN_TEST_RATIO)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        # Normalisation
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df[['close', 'rsi', 'macd', 'atr']])
        test_scaled = scaler.transform(test_df[['close', 'rsi', 'macd', 'atr']])

        # Préparation des séquences
        def create_sequences(data):
            X, y = [], []
            for i in range(LOOKBACK, len(data)):
                X.append(data[i-LOOKBACK:i])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
            
        X_train, y_train = create_sequences(train_scaled)
        X_test, y_test = create_sequences(test_scaled)

        # === VÉRIFICATION DES DONNÉES ===
        if X_train.size == 0 or y_train.size == 0 or X_test.size == 0 or y_test.size == 0:
            await update.message.reply_text("❌ Pas assez de données pour entraîner ou tester le modèle. Essayez avec une autre crypto ou une période plus longue.")
            return ConversationHandler.END

        # Chargement/Entraînement du modèle
        model_path = os.path.join(MODELS_DIR, f'{user_input}_model.keras')
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([
                Input(shape=(X_train.shape[1], X_train.shape[2])),  # Amélioration Keras recommandée
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
            model.save(model_path)

        # Backtesting
        test_predictions = model.predict(X_test)
        test_accuracy = np.mean(np.sign(test_predictions.flatten()) == np.sign(y_test)) * 100

        # Prédiction actuelle
        last_sequence = test_scaled[-LOOKBACK:]
        predicted_price = model.predict(np.array([last_sequence]))[0][0]
        predicted_price = scaler.inverse_transform([[predicted_price, 0, 0, 0]])[0][0]

        # Calcul TP/SL
        current_price = df['close'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        
        if predicted_price > current_price:
            direction = "LONG 🟢"
            tp = current_price + (current_atr * RISK_REWARD_RATIO)
            sl = current_price - current_atr
        else:
            direction = "SHORT 🔴"
            tp = current_price - (current_atr * RISK_REWARD_RATIO)
            sl = current_price + current_atr

        # Visualisation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        df['close'].plot(ax=ax1, title='Prix', color='blue')
        ax1.axhline(tp, color='green', ls='--', label='TP')
        ax1.axhline(sl, color='red', ls='--', label='SL')
        ax1.legend()
        
        df['atr'].plot(ax=ax2, title='Volatilité (ATR)', color='orange')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Message final
        message = (
            f"📊 {user_input.upper()} - Analyse Professionnelle\n\n"
            f"🎯 Direction: {direction}\n"
            f"💰 Prix actuel: {current_price:.2f}$\n"
            f"📈 TP: {tp:.2f}$ (+{(tp/current_price-1)*100:.1f}%)\n"
            f"📉 SL: {sl:.2f}$ ({(sl/current_price-1)*100:.1f}%)\n"
            f"📊 Précision backtest: {test_accuracy:.1f}%\n"
            f"⚡ Volatilité (ATR): {current_atr:.2f}$"
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

def main() -> None:
    # Vérification du token
    if not TELEGRAM_TOKEN:
        raise ValueError("Token Telegram non trouvé. Vérifie le fichier .env.")

    # Configuration des valeurs par défaut pour les messages (optionnel)
    defaults = Defaults(
        parse_mode="HTML",  # ou "Markdown" selon tes besoins
        disable_notification=False
    )
    
    application = Application.builder() \
        .token(TELEGRAM_TOKEN) \
        .defaults(defaults) \
        .build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={ 
            CHOOSING: [MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_token)]
        },
        fallbacks=[]
    )

    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == '__main__':
    main()
