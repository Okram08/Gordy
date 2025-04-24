import os
import logging
import requests
import pandas as pd
import ta
import aiohttp
import asyncio
import time
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes,
)
from functools import lru_cache
from typing import Optional, Tuple

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

# --- Env ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

TOP_TOKENS = [
    ("bitcoin", "BTCUSDT"), ("ethereum", "ETHUSDT"), ("ripple", "XRPUSDT"),
    ("bnb", "BNBUSDT"), ("solana", "SOLUSDT"), ("dogecoin", "DOGEUSDT"),
    ("cardano", "ADAUSDT"), ("tron", "TRXUSDT"), ("sui", "SUIUSDT"),
    ("chainlink", "LINKUSDT"), ("avalanche", "AVAXUSDT"), ("stellar", "XLMUSDT"),
    ("leo", "LEOUSDT"), ("shiba inu", "SHIBUSDT"), ("toncoin", "TONUSDT"),
    ("hedera", "HBARUSDT"), ("bitcoin cash", "BCHUSDT"), ("polkadot", "DOTUSDT"),
    ("litecoin", "LTCUSDT"), ("polygon", "MATICUSDT"),
]

@lru_cache(maxsize=50)
def cached_symbol_data(symbol: str) -> Optional[pd.DataFrame]:
    return asyncio.run(get_binance_ohlc(symbol))

async def get_binance_ohlc(symbol: str, interval: str = "1h", limit: int = 48) -> Optional[pd.DataFrame]:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for _ in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 429:
                        await asyncio.sleep(1)
                        continue
                    response.raise_for_status()
                    data = await response.json()
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
            logger.error(f"Erreur récupération {symbol} : {e}")
            await asyncio.sleep(1)
    return None

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["EMA10"] = ta.trend.ema_indicator(df["close"], window=10)
    df["RSI"] = ta.momentum.rsi(df["close"], window=14)
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    return df

def generate_signal_and_score(df: pd.DataFrame) -> Tuple[str, float, str]:
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
    return signal, score, commentaire

async def classement(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        message = await update.message.reply_text("Analyse des 20 premiers tokens (hors stablecoins)...\n")
    except Exception as e:
        logger.error(f"Erreur lors de la réponse initiale : {e}")
        return

    results = []
    progress_msg = ""
    tasks = []

    for name, symbol in TOP_TOKENS:
        tasks.append(analyse_token(name, symbol))

    tokens_results = await asyncio.gather(*tasks)

    for res in tokens_results:
        progress_msg += res["message"]
        if res["result"]:
            results.append(res["result"])
        try:
            await message.edit_text(progress_msg)
        except Exception:
            pass

    if not results:
        await message.edit_text(progress_msg + "\nAucun signal fort détecté sur les 20 premiers tokens.")
        return

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
    final_msg = progress_msg + "\n🏆 Classement final : Top 3 tokens avec les signaux les plus forts :\n"
    for i, res in enumerate(results, 1):
        final_msg += (
            f"\n{i}. {res['name']} ({res['symbol']})\n"
            f"   Prix : {res['price']:.4f} USDT\n"
            f"   Signal : {res['signal']}\n"
            f"   Score : {res['score']:.2f}\n"
            f"   Commentaire : {res['commentaire']}\n"
        )
    await message.edit_text(final_msg)

async def analyse_token(name: str, symbol: str) -> dict:
    try:
        df = await get_binance_ohlc(symbol)
        if df is None or len(df) < 21:
            return {"message": f"❌ Pas assez de données pour {name.title()} ({symbol}).\n", "result": None}
        df = compute_indicators(df)
        signal, score, commentaire = generate_signal_and_score(df)
        latest = df.iloc[-1]
        if signal == "HOLD":
            return {"message": f"➖ Aucun signal fort pour {name.title()} ({symbol}).\n", "result": None}
        return {
            "message": (
                f"\n✅ {name.title()} ({symbol}):\n"
                f"Prix : {latest['close']:.4f} USDT\n"
                f"EMA10 : {latest['EMA10']:.4f} | SMA20 : {latest['SMA20']:.4f}\n"
                f"RSI : {latest['RSI']:.2f}\n"
                f"Bollinger : [{latest['BB_lower']:.4f} ; {latest['BB_upper']:.4f}]\n"
                f"Signal : {signal} | Score : {score:.2f}\n"
                f"{commentaire}\n"
            ),
            "result": {
                "name": name.title(), "symbol": symbol,
                "signal": signal, "score": score,
                "price": latest["close"], "commentaire": commentaire
            }
        }
    except Exception as e:
        logger.error(f"Erreur analyse {name} : {e}")
        return {"message": f"⚠️ Erreur analyse {name.title()} ({symbol}) : {e}\n", "result": None}

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("classement", classement))
    logger.info("Bot lancé et en attente de commandes.")
    app.run_polling()
