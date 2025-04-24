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
    if df["close"].isnull().all() or df["close"].max() == df["close"].min():
        raise ValueError("Les données sont insuffisantes ou trop similaires pour l'analyse technique.")

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
            score += abs(latest["EMA10"] - latest["SMA20"])
            score += max(0, 35 - latest["RSI"])
            score += latest["MACD"]
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
            score += abs(latest["EMA10"] - latest["SMA20"])
            score += max(0, latest["RSI"] - 65)
            score += abs(latest["MACD"])
            score += latest["ADX"] / 2
            commentaire = "Signal de vente confirmé (surachat, volume élevé, MACD baissier, tendance forte)."
        else:
            signal = "🤝 HOLD"
            commentaire = "Aucun signal de vente fort malgré la tendance baissière."
    else:
        signal = "🤝 HOLD"
        commentaire = "Aucun signal clair détecté."
    return signal, score, commentaire

# --- Menu retour ---
def menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Nouvelle Analyse", callback_data="analyse")],
        [InlineKeyboardButton("🏆 Voir Classement", callback_data="classement")]
    ])

# --- Commandes Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Bienvenue ! Utilisez les boutons ci-dessous pour commencer.",
        reply_markup=menu_keyboard()
    )

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "analyse":
        keyboard = [
            [InlineKeyboardButton(name.title(), callback_data=symbol)]
            for name, symbol in TOP_TOKENS
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text("📊 Sélectionnez une crypto :", reply_markup=reply_markup)

    elif query.data == "classement":
        await classement(query, context)

    else:
        symbol = query.data
        name = next((name.title() for name, sym in TOP_TOKENS if sym == symbol), symbol)
        await query.edit_message_text(text=f"🔍 Analyse de {name} en cours...")

        df = get_binance_ohlc(symbol)
        if df is None or len(df) < 50:
            await query.edit_message_text(f"❌ Pas assez de données pour {name}.", reply_markup=menu_keyboard())
            return

        try:
            df = compute_indicators(df)
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs pour {symbol}: {e}")
            await query.edit_message_text(f"❌ Analyse impossible pour {name} (erreur: {e}).", reply_markup=menu_keyboard())
            return

        signal, score, commentaire = generate_signal_and_score(df)
        latest = df.iloc[-1]

        result = (
            f"📊 Résultat pour {name} ({symbol}):\n"
            f"Prix : {latest['close']:.4f} USDT\n"
            f"EMA10 : {latest['EMA10']:.4f} | SMA20 : {latest['SMA20']:.4f} | SMA200 : {latest['SMA200']:.4f}\n"
            f"RSI : {latest['RSI']:.2f} | MACD : {latest['MACD']:.4f} | ADX : {latest['ADX']:.2f}\n"
            f"Bollinger : [{latest['BB_lower']:.4f} ; {latest['BB_upper']:.4f}]\n"
            f"Volume actuel : {latest['volume']:.2f} | Moyenne : {latest['volume_mean']:.2f}\n"
            f"Signal : {signal} | Score : {score:.2f}\n"
            f"{commentaire}"
        )
        await query.edit_message_text(result, reply_markup=menu_keyboard())

async def classement(query, context):
    await query.edit_message_text("📈 Classement en cours...")
    results = []

    for name, symbol in TOP_TOKENS:
        df = get_binance_ohlc(symbol)
        if df is None or len(df) < 50:
            continue
        try:
            df = compute_indicators(df)
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs pour {symbol}: {e}")
            continue
        _, score, _ = generate_signal_and_score(df)
        results.append((name.title(), symbol, score))

    results.sort(key=lambda x: x[2], reverse=True)
    classement_text = "🏆 Top 10 cryptos avec meilleur score:\n\n"
    for i, (name, symbol, score) in enumerate(results[:10], 1):
        classement_text += f"{i}. {name} ({symbol}) — Score: {score:.2f}\n"

    await query.edit_message_text(classement_text, reply_markup=menu_keyboard())
