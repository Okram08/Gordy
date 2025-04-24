import os
import pandas as pd
from dotenv import load_dotenv
from pycoingecko import CoinGeckoAPI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import ta

# Charger les variables d'environnement
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")  # Optionnel

# Initialiser CoinGecko API
cg = CoinGeckoAPI(api_key=COINGECKO_API_KEY) if COINGECKO_API_KEY else CoinGeckoAPI()

def get_price_history(symbol="bitcoin", vs_currency="usd", days=2):
    # Récupère les prix horaires sur 2 jours
    data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency=vs_currency, days=days, interval="hourly")
    prices = [x[1] for x in data["prices"]]
    timestamps = [x[0] for x in data["prices"]]
    df = pd.DataFrame({"close": prices}, index=pd.to_datetime(timestamps, unit='ms'))
    return df

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
    # Stratégie simple : convergence de signaux
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

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Permet de passer le symbole en argument : /signal bitcoin
    symbol = "bitcoin"
    if context.args:
        symbol = context.args[0].lower()
    try:
        df = get_price_history(symbol)
        df = compute_indicators(df)
        signal = generate_signal(df)
        last_price = df["close"].iloc[-1]
        await update.message.reply_text(
            f"Crypto : {symbol.capitalize()}\n"
            f"Prix actuel : {last_price:.2f} USD\n"
            f"Signal sur 24h : {signal}"
        )
    except Exception as e:
        await update.message.reply_text(f"Erreur : {str(e)}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bienvenue ! Utilisez la commande /signal <crypto> pour recevoir un signal de trading sur 24h.\nExemple : /signal bitcoin")

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))
    print("Bot lancé !")
    app.run_polling()
