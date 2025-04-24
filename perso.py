import os
import logging
import requests
import pandas as pd
import ta
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes,
    ConversationHandler, filters
)

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

# --- Env ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# --- Conversation state ---
ASK_TOKEN = 1

# --- Utilitaires Binance ---
def get_binance_symbols():
    """Récupère tous les symboles de trading disponibles sur Binance."""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        symbols = data["symbols"]
        # Crée un mapping {baseAsset.lower(): symbol} pour les paires contre USDT
        mapping = {s["baseAsset"].lower(): s["symbol"] for s in symbols if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}
        return mapping
    except Exception as e:
        logger.error(f"Erreur récupération des symboles Binance : {e}")
        return {}

def get_binance_ohlc(symbol, interval="1h", limit=48):
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
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    return df

def generate_signal(df):
    latest = df.iloc[-1]
    if (
        latest["EMA10"] > latest["SMA20"] and
        latest["RSI"] < 30 and
        latest["close"] < latest["BB_lower"]
    ):
        return "BUY"
    elif (
        latest["EMA10"] < latest["SMA20"] and
        latest["RSI"] > 70 and
        latest["close"] > latest["BB_upper"]
    ):
        return "SELL"
    else:
        return "HOLD"

# --- Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} started the bot.")
    await update.message.reply_text(
        "Bienvenue ! Envoie-moi le nom du token (ex: bitcoin, ethereum, solana, pepe...) que tu veux analyser."
    )
    return ASK_TOKEN

async def ask_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    token_name = update.message.text.strip().lower()
    logger.info(f"User {update.effective_user.id} demande analyse pour: {token_name}")

    # Récupère la liste des symboles Binance (cache en mémoire pour éviter trop de requêtes)
    if not hasattr(ask_token, "symbol_mapping"):
        ask_token.symbol_mapping = get_binance_symbols()

    symbol_mapping = ask_token.symbol_mapping

    if token_name not in symbol_mapping:
        await update.message.reply_text(
            f"Token '{token_name}' non trouvé ou non disponible contre USDT sur Binance.\n"
            "Essaie par exemple : bitcoin, ethereum, solana, pepe, dogecoin, etc."
        )
        logger.warning(f"Token non trouvé : {token_name}")
        return ConversationHandler.END

    symbol = symbol_mapping[token_name]
    df = get_binance_ohlc(symbol)
    if df is None or len(df) < 21:
        await update.message.reply_text(
            f"Impossible de récupérer suffisamment de données pour {token_name.upper()} ({symbol})."
        )
        logger.warning(f"Echec récupération données pour {symbol}")
        return ConversationHandler.END

    df = compute_indicators(df)
    signal = generate_signal(df)
    latest = df.iloc[-1]
    await update.message.reply_text(
        f"Analyse pour '{token_name.upper()}' ({symbol}):\n"
        f"Prix actuel : {latest['close']:.4f} USDT\n"
        f"EMA10 : {latest['EMA10']:.4f}\n"
        f"SMA20 : {latest['SMA20']:.4f}\n"
        f"RSI : {latest['RSI']:.2f}\n"
        f"Bollinger Lower : {latest['BB_lower']:.4f}\n"
        f"Bollinger Upper : {latest['BB_upper']:.4f}\n"
        f"Signal sur 24h : {signal}"
    )
    logger.info(f"Analyse envoyée pour {token_name.upper()} - Signal: {signal}")
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Opération annulée.")
    logger.info(f"User {update.effective_user.id} a annulé l'opération.")
    return ConversationHandler.END

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_TOKEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_token)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(conv_handler)
    logger.info("Bot lancé et en attente de commandes.")
    app.run_polling()
