import os
import logging
import requests
import pandas as pd
import ta
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
)
import asyncio

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

# --- Env ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Liste des 20 premiers tokens hors stablecoins (2025)
TOP_TOKENS = [
    ("bitcoin", "BTCUSDT"),
    ("ethereum", "ETHUSDT"),
    ("ripple", "XRPUSDT"),
    ("bnb", "BNBUSDT"),
    ("solana", "SOLUSDT"),
    ("dogecoin", "DOGEUSDT"),
    ("cardano", "ADAUSDT"),
    ("tron", "TRXUSDT"),
    ("sui", "SUIUSDT"),
    ("chainlink", "LINKUSDT"),
    ("avalanche", "AVAXUSDT"),
    ("stellar", "XLMUSDT"),
    ("leo", "LEOUSDT"),
    ("shiba inu", "SHIBUSDT"),
    ("toncoin", "TONUSDT"),
    ("hedera", "HBARUSDT"),
    ("bitcoin cash", "BCHUSDT"),
    ("polkadot", "DOTUSDT"),
    ("litecoin", "LTCUSDT"),
    ("polygon", "MATICUSDT"),
]

# --- Analyse ---
def get_binance_ohlc(symbol, interval="1h", limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[ 
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.set_index("open_time", inplace=True)
        return df
    except Exception as e:
        logger.error(f"Erreur récupération données Binance : {e}")
        return None

def compute_indicators(df):
    df["SMA20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["EMA10"] = ta.trend.ema_indicator(df["close"], window=10)
    df["RSI"] = ta.momentum.rsi(df["close"], window=14)
    df["SMA200"] = ta.trend.sma_indicator(df["close"], window=200)
    df["MACD"] = ta.trend.macd_diff(df["close"])
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["volume_mean"] = df["volume"].rolling(window=20).mean()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    return df

def generate_signal_and_score(df):
    latest = df.iloc[-1]
    score = 0
    commentaire = ""

    # Calcul du score ajusté
    if latest["close"] > latest["SMA200"]:
        if (
            latest["EMA10"] > latest["SMA20"] and
            latest["RSI"] < 35 and
            latest["close"] < latest["BB_lower"] and
            latest["MACD"] > 0 and
            latest["ADX"] > 20 and
            latest["volume"] > 1.5 * latest["volume_mean"]
        ):
            signal = "📈 BUY"
            score += abs(latest["EMA10"] - latest["SMA20"]) * 2  # Poids accru pour la tendance
            score += max(0, 35 - latest["RSI"]) * 2  # Poids accru pour RSI
            score += latest["MACD"] * 1.5  # Poids pour MACD
            score += latest["ADX"] / 2
            commentaire = "Signal d'achat confirmé (survente, volume élevé, MACD haussier, tendance forte)."
        else:
            signal = "🤝 HOLD"
            commentaire = "Aucun signal d'achat fort malgré la tendance haussière."
    elif latest["close"] < latest["SMA200"]:
        if (
            latest["EMA10"] < latest["SMA20"] and
            latest["RSI"] > 65 and
            latest["close"] > latest["BB_upper"] and
            latest["MACD"] < 0 and
            latest["ADX"] > 20 and
            latest["volume"] > 1.5 * latest["volume_mean"]
        ):
            signal = "📉 SELL"
            score += abs(latest["EMA10"] - latest["SMA20"]) * 2
            score += max(0, latest["RSI"] - 65) * 2
            score += abs(latest["MACD"]) * 1.5
            score += latest["ADX"] / 2
            commentaire = "Signal de vente confirmé (surachat, volume élevé, MACD baissier, tendance forte)."
        else:
            signal = "🤝 HOLD"
            commentaire = "Aucun signal de vente fort malgré la tendance baissière."
    else:
        signal = "🤝 HOLD"
        commentaire = "Aucun signal clair détecté."
    return signal, score, commentaire

# --- Backtesting simple ---
def backtest_strategy(df, initial_balance=1000):
    balance = initial_balance
    position = 0
    for idx in range(1, len(df)):
        signal, score, _ = generate_signal_and_score(df.iloc[:idx])
        if signal == "📈 BUY" and balance > 0:
            position = balance / df.iloc[idx]["close"]
            balance = 0
        elif signal == "📉 SELL" and position > 0:
            balance = position * df.iloc[idx]["close"]
            position = 0
    if position > 0:
        balance = position * df.iloc[-1]["close"]
    return balance

# --- Gestion du risque ---
def manage_risk(signal, price, stop_loss_pct=0.03, take_profit_pct=0.05):
    stop_loss = price * (1 - stop_loss_pct) if signal == "📈 BUY" else price * (1 + stop_loss_pct)
    take_profit = price * (1 + take_profit_pct) if signal == "📈 BUY" else price * (1 - take_profit_pct)
    return stop_loss, take_profit

# --- Ecran d'accueil ---
async def accueil(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🏆 Classement", callback_data="menu_classement")],
        [InlineKeyboardButton("📊 Analyse", callback_data="menu_analyse")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    chat_id = update.effective_chat.id if update.effective_chat else update.callback_query.message.chat_id
    await context.bot.send_message(
        chat_id=chat_id,
        text="👋 Bienvenue ! Que souhaitez-vous faire ?",
        reply_markup=reply_markup
    )

# --- Commande /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await accueil(update, context)

# --- Main ---
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(menu_handler, pattern="^menu_"))
    app.add_handler(CallbackQueryHandler(analyse_token_callback, pattern="^analyse_"))
    logger.info("Bot lancé et en attente de commandes.")
    app.run_polling()
