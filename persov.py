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

# --- Ecran d'accueil ---
async def accueil(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Créer un clavier inline avec les boutons
    keyboard = [
        [InlineKeyboardButton("🏆 Classement", callback_data="menu_classement")],
        [InlineKeyboardButton("📊 Analyse", callback_data="menu_analyse")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Récupérer l'ID du chat
    chat_id = (
        update.effective_chat.id
        if update.effective_chat
        else update.callback_query.message.chat_id
    )
    
    # Envoyer un message avec un clavier inline, désactivant la barre de message
    await context.bot.send_message(
        chat_id=chat_id,
        text="👋 Bienvenue ! Que souhaitez-vous faire ?",
        reply_markup=reply_markup
    )
    
    # Désactivation de la barre de message via un nouveau message sans `reply_markup` :
    await context.bot.send_message(
        chat_id=chat_id,
        text="Vous ne pouvez plus envoyer de messages libres. Veuillez utiliser les boutons ci-dessus.",
        reply_markup=None
    )

# --- Commande /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await accueil(update, context)

# --- Handler du menu principal ---
async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data

    if data == "menu_classement":
        await classement_callback(update, context)
    elif data == "menu_analyse":
        await analyse_callback(update, context)

# --- Classement (callback) ---
async def classement_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = query.message.chat_id
    message = await context.bot.send_message(chat_id=chat_id, text="🔎 Analyse des 20 tokens en cours...")
    results = []
    progress_msg = ""

    for idx, (name, symbol) in enumerate(TOP_TOKENS, 1):
        try:
            df = get_binance_ohlc(symbol)
            if df is None or len(df) < 50:
                progress_msg += f"❌ {name.title()} : Données insuffisantes.\n"
                await context.bot.edit_message_text(progress_msg, chat_id=chat_id, message_id=message.message_id)
                continue

            df = compute_indicators(df)
            signal, score, commentaire = generate_signal_and_score(df)
            latest = df.iloc[-1]

            if signal != "🤝 HOLD":
                results.append({
                    "name": name.title(),
                    "symbol": symbol,
                    "signal": signal,
                    "score": score,
                    "price": latest["close"],
                    "commentaire": commentaire
                })
                progress_msg += f"✅ {name.title()} : Signal {signal} | Score {score:.2f}\n"
            else:
                progress_msg += f"➖ {name.title()} : Aucun signal fort.\n"

            await context.bot.edit_message_text(progress_msg, chat_id=chat_id, message_id=message.message_id)
            await asyncio.sleep(0.3)
        except Exception as e:
            logger.error(f"Erreur analyse {name}: {e}")
            progress_msg += f"⚠️ {name.title()} : erreur pendant l'analyse.\n"
            await context.bot.edit_message_text(progress_msg, chat_id=chat_id, message_id=message.message_id)

    if not results:
        await context.bot.edit_message_text("Aucun signal fort détecté.", chat_id=chat_id, message_id=message.message_id)
        # Affiche l'écran d'accueil dans un NOUVEAU message
        await accueil(update, context)
        return

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
    final_msg = progress_msg + "\n🏆 Top 3 tokens avec les signaux les plus forts :\n"
    for i, res in enumerate(results, 1):
        final_msg += (
            f"\n{i}. {res['name']} ({res['symbol']})\n"
            f"   Prix : {res['price']:.4f} USDT\n"
            f"   Signal : {res['signal']} | Score : {res['score']:.2f}\n"
            f"   Commentaire : {res['commentaire']}\n"
        )
    await context.bot.edit_message_text(final_msg, chat_id=chat_id, message_id=message.message_id)
    # Affiche l'écran d'accueil dans un NOUVEAU message
    await accueil(update, context)

# --- Analyse (callback) ---
async def analyse_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton(name.title(), callback_data=f"analyse_{symbol}")]
        for name, symbol in TOP_TOKENS
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    chat_id = update.callback_query.message.chat_id
    await context.bot.send_message(chat_id=chat_id, text="📊 Sélectionnez une crypto :", reply_markup=reply_markup)

# --- Analyse d'un token (callback) ---
async def analyse_token_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = query.message.chat_id
    symbol = query.data.replace("analyse_", "")
    name = next((name.title() for name, sym in TOP_TOKENS if sym == symbol), symbol)
    await context.bot.send_message(chat_id=chat_id, text=f"🔍 Analyse de {name} en cours...")

    df = get_binance_ohlc(symbol)
    if df is None or len(df) < 50:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Pas assez de données pour {name}.")
        await accueil(update, context)
        return

    df = compute_indicators(df)
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
    await context.bot.send_message(chat_id=chat_id, text=result)
    # Affiche l'écran d'accueil dans un NOUVEAU message
    await accueil(update, context)

# --- Main ---
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(menu_handler, pattern="^menu_"))
    app.add_handler(CallbackQueryHandler(analyse_token_callback, pattern="^analyse_"))
    logger.info("Bot lancé et en attente de commandes.")
    app.run_polling()
