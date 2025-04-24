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
    commentaire = ""
    if (
        latest["EMA10"] > latest["SMA20"] and
        latest["RSI"] < 35 and
        latest["close"] < latest["BB_lower"]
    ):
        signal = "BUY"
        score += abs(latest["EMA10"] - latest["SMA20"])
        score += max(0, 35 - latest["RSI"])
        commentaire = "Forte probabilité de rebond technique (survente et tendance haussière naissante)."
    elif (
        latest["EMA10"] < latest["SMA20"] and
        latest["RSI"] > 65 and
        latest["close"] > latest["BB_upper"]
    ):
        signal = "SELL"
        score += abs(latest["EMA10"] - latest["SMA20"])
        score += max(0, latest["RSI"] - 65)
        commentaire = "Risque de correction (surachat et tendance baissière amorcée)."
    else:
        signal = "HOLD"
        commentaire = "Aucun signal fort détecté."
        score = 0
    return signal, score, commentaire

async def classement(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyse des 20 premiers tokens (hors stablecoins)...\n")
    results = []
    for name, symbol in TOP_TOKENS:
        try:
            await update.message.reply_text(f"Analyse de {name.title()} ({symbol}) en cours...")
            df = get_binance_ohlc(symbol)
            if df is None or len(df) < 21:
                await update.message.reply_text(f"Pas assez de données pour {name.title()} ({symbol}).")
                continue
            df = compute_indicators(df)
            signal, score, commentaire = generate_signal_and_score(df)
            latest = df.iloc[-1]
            if signal != "HOLD":
                msg = (
                    f"Résultat pour {name.title()} ({symbol}):\n"
                    f"Prix actuel : {latest['close']:.4f} USDT\n"
                    f"EMA10 : {latest['EMA10']:.4f}\n"
                    f"SMA20 : {latest['SMA20']:.4f}\n"
                    f"RSI : {latest['RSI']:.2f}\n"
                    f"Bollinger Lower : {latest['BB_lower']:.4f}\n"
                    f"Bollinger Upper : {latest['BB_upper']:.4f}\n"
                    f"Signal : {signal}\n"
                    f"Score de confiance : {score:.2f}\n"
                    f"Commentaire : {commentaire}\n"
                )
                await update.message.reply_text(msg)
                results.append({
                    "name": name.title(),
                    "symbol": symbol,
                    "signal": signal,
                    "score": score,
                    "price": latest["close"],
                    "commentaire": commentaire
                })
            else:
                await update.message.reply_text(f"Aucun signal fort détecté pour {name.title()} ({symbol}).")
        except Exception as e:
            await update.message.reply_text(f"Erreur lors de l'analyse de {name.title()} ({symbol}) : {e}")
            logger.error(f"Erreur analyse {name} : {e}")

    if not results:
        await update.message.reply_text("Aucun signal fort détecté sur les 20 premiers tokens.")
        return

    # Classement par score décroissant
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top3 = results[:3]
    msg = "🏆 Classement final : Top 3 tokens avec les signaux les plus forts :\n"
    for i, res in enumerate(top3, 1):
        msg += (
            f"\n{i}. {res['name']} ({res['symbol']})\n"
            f"   Prix : {res['price']:.4f} USDT\n"
            f"   Signal : {res['signal']}\n"
            f"   Score : {res['score']:.2f}\n"
            f"   Commentaire : {res['commentaire']}\n"
        )
    await update.message.reply_text(msg)

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("classement", classement))
    logger.info("Bot lancé et en attente de commandes.")
    app.run_polling()
