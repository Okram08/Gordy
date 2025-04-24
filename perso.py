import os
import logging
import requests
import pandas as pd
import ta
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes,
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

def generate_signal_and_score(df):
    latest = df.iloc[-1]
    score = 0
    # Signal
    if (
        latest["EMA10"] > latest["SMA20"] and
        latest["RSI"] < 35 and
        latest["close"] < latest["BB_lower"]
    ):
        signal = "BUY"
        score += abs(latest["EMA10"] - latest["SMA20"])
        score += max(0, 35 - latest["RSI"])
    elif (
        latest["EMA10"] < latest["SMA20"] and
        latest["RSI"] > 65 and
        latest["close"] > latest["BB_upper"]
    ):
        signal = "SELL"
        score += abs(latest["EMA10"] - latest["SMA20"])
        score += max(0, latest["RSI"] - 65)
    else:
        signal = "HOLD"
        score = 0
    return signal, score

async def classement(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyse des 20 premiers tokens (hors stablecoins)...")
    results = []
    for name, symbol in TOP_TOKENS:
        df = get_binance_ohlc(symbol)
        if df is None or len(df) < 21:
            continue
        df = compute_indicators(df)
        signal, score = generate_signal_and_score(df)
        if signal != "HOLD":
            results.append({
                "name": name.title(),
                "symbol": symbol,
                "signal": signal,
                "score": score,
                "price": df.iloc[-1]["close"]
            })
    if not results:
        await update.message.reply_text("Aucun signal fort détecté sur les 20 premiers tokens.")
        return
    # Classement par score décroissant
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top3 = results[:3]
    msg = "Top 3 tokens avec les signaux les plus forts :\n"
    for i, res in enumerate(top3, 1):
        msg += (
            f"\n{i}. {res['name']} ({res['symbol']})\n"
            f"   Prix : {res['price']:.4f} USDT\n"
            f"   Signal : {res['signal']}\n"
            f"   Score : {res['score']:.2f}\n"
        )
    await update.message.reply_text(msg)

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("classement", classement))
    logger.info("Bot lancé et en attente de commandes.")
    app.run_polling()
