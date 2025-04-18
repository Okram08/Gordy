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
    ContextTypes,
    Defaults
)
from pycoingecko import CoinGeckoAPI
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import pandas_ta as ta

# Configuration initiale
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

cg = CoinGeckoAPI()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Hyperparamètres
LOOKBACK = 12
ATR_PERIOD = 14
RISK_REWARD_RATIO = 2
TRAIN_TEST_RATIO = 0.8

@lru_cache(maxsize=100)
def get_crypto_data(token: str, days: int):
    try:
        return cg.get_coin_ohlc_by_id(id=token, vs_currency='usd', days=days)
    except Exception as e:
        logging.error(f"API Error for {token}: {str(e)}")
        return None

def compute_macd(data):
    short_ema = data.ewm(span=12, adjust=False).mean()
    long_ema = data.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def compute_atr(high, low, close):
    return ta.atr(high, low, close, length=14)

def detect_market_conditions(df):
    """Détecte la volatilité et la tendance"""
    df['atr'] = compute_atr(df['high'], df['low'], df['close'])
    volatilite = df['atr'].iloc[-1] / df['close'].iloc[-1]
    
    adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
    adx = adx_data['ADX_14'].iloc[-1]
    plus_di = adx_data['DMP_14'].iloc[-1]
    minus_di = adx_data['DMN_14'].iloc[-1]
    
    tendance = "neutre"
    if adx > 25:
        tendance = "haussière" if plus_di > minus_di else "baissière"
    
    return volatilite, tendance

def optimize_rsi_period(df):
    volatilite, tendance = detect_market_conditions(df)
    if tendance != "neutre" and volatilite < 0.02:
        return 21
    elif volatilite > 0.05:
        return 9
    else:
        return 14

def compute_rsi(data, period):
    return ta.rsi(data, length=period)

def optimize_data_period(token: str) -> int:
    """Détermine la période de données selon la capitalisation"""
    try:
        coin_data = cg.get_coin_by_id(token)
        market_cap = coin_data['market_data']['market_cap']['usd']
        return 365 if market_cap > 1e10 else 90
    except:
        return 90  # Fallback

async def analyze_and_reply(update: Update, token: str):
    """Effectue toute l'analyse et répond à l'utilisateur"""
    days = optimize_data_period(token)
    await update.message.reply_text(f"🔍 Analyse en cours pour {token} (période auto: {days}j)...")
    
    try:
        ohlc = get_crypto_data(token, days)
        if not ohlc:
            await update.message.reply_text("❌ Cryptomonnaie non trouvée")
            return

        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Indicateurs techniques
        df['macd'], df['signal'] = compute_macd(df['close'])
        df['atr'] = compute_atr(df['high'], df['low'], df['close'])
        rsi_period = optimize_rsi_period(df)
        df['rsi'] = compute_rsi(df['close'], rsi_period)
        df.dropna(inplace=True)

        if len(df) < LOOKBACK * 2:
            await update.message.reply_text("❌ Données insuffisantes")
            return

        # Préparation des données
        train_size = int(len(df) * TRAIN_TEST_RATIO)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df[['close', 'rsi', 'macd', 'atr']])
        test_scaled = scaler.transform(test_df[['close', 'rsi', 'macd', 'atr']])

        def create_sequences(data):
            X, y = [], []
            for i in range(LOOKBACK, len(data)):
                X.append(data[i-LOOKBACK:i])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
            
        X_train, y_train = create_sequences(train_scaled)
        X_test, y_test = create_sequences(test_scaled)

        # Modèle LSTM
        model_path = os.path.join(MODELS_DIR, f'{token}_model.keras')
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([
                Input(shape=(X_train.shape[1], X_train.shape[2])),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            model.save(model_path)

        # Prédiction
        last_sequence = test_scaled[-LOOKBACK:]
        predicted_price = model.predict(np.array([last_sequence]))[0][0]
        predicted_price = scaler.inverse_transform([[predicted_price, 0, 0, 0]])[0][0]
        current_price = df['close'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        
        direction = "LONG 🟢" if predicted_price > current_price else "SHORT 🔴"
        tp = current_price + (current_atr * RISK_REWARD_RATIO) if direction.startswith("LONG") else current_price - (current_atr * RISK_REWARD_RATIO)
        sl = current_price - current_atr if direction.startswith("LONG") else current_price + current_atr

        # Visualisation
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        df['close'].plot(ax=axs[0], title='Prix', color='blue')
        axs[0].axhline(tp, color='green', ls='--', label='TP')
        axs[0].axhline(sl, color='red', ls='--', label='SL')
        axs[0].legend()
        
        df['rsi'].plot(ax=axs[1], color='purple', title=f'RSI ({rsi_period} périodes)')
        axs[1].axhline(70, color='red', ls='--', alpha=0.5)
        axs[1].axhline(30, color='green', ls='--', alpha=0.5)
        
        df[['macd', 'signal']].plot(ax=axs[2], color=['orange', 'black'], title='MACD')
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Message final
message = (
    f"📊 {token.upper()} - Analyse automatisée\n"
    f"📅 Période données: {days}j | RSI: {rsi_period}p\n"
    f"🎯 {direction}\n"
    f"💰 Prix: {current_price:.2f}$\n"  # Guillemet ajouté après f
    f"📈 TP: {tp:.2f}$ | 📉 SL: {sl:.2f}$\n"
    f"⚡ ATR: {current_atr:.2f}$"
)


        await update.message.reply_photo(
            photo=InputFile(buf, filename='analysis.png'),
            caption=message
        )
        buf.close()

    except Exception as e:
        logging.error(f"Erreur: {str(e)}")
        await update.message.reply_text("❌ Une erreur est survenue durant l'analyse")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère la commande /start"""
    if not context.args:
        await update.message.reply_text("❌ Usage: /start <crypto> (ex: /start bitcoin)")
        return
    
    token = context.args[0].lower()
    await analyze_and_reply(update, token)

def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.run_polling()

if __name__ == '__main__':
    main()
