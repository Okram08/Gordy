import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes,
    ConversationHandler, filters
)
from pycoingecko import CoinGeckoAPI
import pandas as pd
import ta

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

# --- Chargement des variables d'environnement ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# --- États de la conversation ---
ASK_TOKEN = 1

# --- Initialisation CoinGecko ---
cg = CoinGeckoAPI()

def get_price_and_history(user_input):
    user_input = user_input.strip().lower()
    # Si c'est une adresse de contrat ERC20 Ethereum
    if user_input.startswith("0x") and len(user_input) == 42:
        # Récupère l'historique sur 2 jours (48h) en hourly
        data = cg.get_coin_market_chart_from_contract_address_by_id(
            id="ethereum",
            contract_address=user_input,
            vs_currency="usd",
            days=2,
            interval="hourly"
        )
        prices = [x[1] for x in data["prices"]]
        timestamps = [x[0] for x in data["prices"]]
        df = pd.DataFrame({"close": prices}, index=pd.to_datetime(timestamps, unit='ms'))
        # Prix actuel
        last_price = prices[-1] if prices else None
        return last_price, df
    else:
        # Crypto native
        data = cg.get_coin_market_chart_by_id(
            id=user_input,
            vs_currency="usd",
            days=2,
            interval="hourly"
        )
        prices = [x[1] for x in data["prices"]]
        timestamps = [x[0] for x in data["prices"]]
        df = pd.DataFrame({"close": prices}, index=pd.to_datetime(timestamps, unit='ms'))
        last_price = prices[-1] if prices else None
        return last_price, df

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

# --- Handlers Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} started the bot.")
    await update.message.reply_text(
        "Bienvenue ! Envoie-moi le nom d'une crypto (ex: bitcoin, ethereum) ou l'adresse du token ERC20 (ex: 0x...) que tu veux analyser."
    )
    return ASK_TOKEN

async def ask_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    logger.info(f"User {update.effective_user.id} demande analyse pour: {user_input}")
    try:
        last_price, df = get_price_and_history(user_input)
        if last_price is None or df is None or len(df) < 21:
            await update.message.reply_text(
                "Impossible de récupérer suffisamment de données pour cette entrée. "
                "Envoie un nom valide (ex: bitcoin, ethereum) ou une adresse de contrat ERC20 (ex: 0x...)."
            )
            logger.warning(f"Echec récupération données pour {user_input}")
            return ConversationHandler.END

        df = compute_indicators(df)
        signal = generate_signal(df)
        latest = df.iloc[-1]
        await update.message.reply_text(
            f"Analyse pour '{user_input}':\n"
            f"Prix actuel : {last_price:.4f} USD\n"
            f"EMA10 : {latest['EMA10']:.4f}\n"
            f"SMA20 : {latest['SMA20']:.4f}\n"
            f"RSI : {latest['RSI']:.2f}\n"
            f"Bollinger Lower : {latest['BB_lower']:.4f}\n"
            f"Bollinger Upper : {latest['BB_upper']:.4f}\n"
            f"Signal sur 24h : {signal}"
        )
        logger.info(f"Analyse envoyée pour {user_input} - Signal: {signal}")
    except Exception as e:
        await update.message.reply_text(
            f"Erreur lors de l'analyse : {e}\n"
            "Envoie un nom valide (ex: bitcoin, ethereum) ou une adresse de contrat ERC20 (ex: 0x...)."
        )
        logger.error(f"Erreur analyse pour {user_input}: {e}")
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
