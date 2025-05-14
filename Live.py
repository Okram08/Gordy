import os
import logging
import requests
import pandas as pd
import ta
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
)
import asyncio

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# --- Env ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# --- Cryptos analysées ---
TOP_TOKENS = [
    ("bitcoin", "BTCUSDT"),
    ("ethereum", "ETHUSDT"),
    ("solana", "SOLUSDT"),
    ("dogecoin", "DOGEUSDT"),
    ("cardano", "ADAUSDT"),
    ("sonic", "SUSDT"),
    ("aave", "AAVEUSDT"),
    ("virtual", "VIRTUALUSDT"),
]

# --- Fonctions données et indicateurs ---
def get_binance_ohlc(symbol, interval="1h", limit=250):
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
    df = df.copy()
    for col in ["close", "high", "low", "volume"]:
        df[col] = df[col].ffill().bfill()
    df["SMA20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["EMA10"] = ta.trend.ema_indicator(df["close"], window=10)
    df["RSI"] = ta.momentum.rsi(df["close"], window=14)
    df["SMA200"] = df["close"].rolling(window=200, min_periods=1).mean()
    df["MACD"] = ta.trend.macd_diff(df["close"])
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["volume_mean"] = df["volume"].rolling(window=20, min_periods=1).mean()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    for col in ["SMA20", "EMA10", "RSI", "SMA200", "MACD", "ADX", "volume_mean", "BB_upper", "BB_lower"]:
        df[col] = df[col].ffill().bfill().fillna(df["close"])
    return df

def generate_signal_and_score(df):
    latest = df.iloc[-1]
    for key in ["SMA200", "BB_upper", "BB_lower"]:
        if pd.isna(latest[key]):
            latest[key] = latest["close"]

    score_buy = 0
    score_sell = 0

    # BUY
    if latest["close"] > latest["SMA200"]: score_buy += 1
    if latest["EMA10"] > latest["SMA20"]: score_buy += 1
    if latest["RSI"] < 40: score_buy += 1
    if latest["close"] < latest["BB_lower"]: score_buy += 1
    if latest["MACD"] > 0: score_buy += 1
    if latest["ADX"] > 20: score_buy += 1
    if latest["volume"] > 1.2 * latest["volume_mean"]: score_buy += 1

    # SELL
    if latest["close"] < latest["SMA200"]: score_sell += 1
    if latest["EMA10"] < latest["SMA20"]: score_sell += 1
    if latest["RSI"] > 60: score_sell += 1
    if latest["close"] > latest["BB_upper"]: score_sell += 1
    if latest["MACD"] < 0: score_sell += 1
    if latest["ADX"] > 20: score_sell += 1
    if latest["volume"] > 1.2 * latest["volume_mean"]: score_sell += 1

    if score_buy >= 4 and score_buy >= score_sell:
        signal = "📈 BUY"
        score = score_buy
        commentaire = f"Signal d'achat ({score_buy}/7 critères validés)."
        confiance = int((score_buy / 7) * 10)
        confiance_txt = (
            "Forte" if score_buy >= 6 else
            "Bonne" if score_buy == 5 else
            "Moyenne"
        )
        entry = latest["close"]
        stop_loss = min(latest["SMA200"] * 0.997, entry * 0.97)
        take_profit = max(latest["BB_upper"] * 0.995, entry * 1.06)
    elif score_sell >= 4 and score_sell > score_buy:
        signal = "📉 SELL"
        score = score_sell
        commentaire = f"Signal de vente ({score_sell}/7 critères validés)."
        confiance = int((score_sell / 7) * 10)
        confiance_txt = (
            "Forte" if score_sell >= 6 else
            "Bonne" if score_sell == 5 else
            "Moyenne"
        )
        entry = latest["close"]
        stop_loss = max(latest["SMA200"] * 1.003, entry * 1.03)
        take_profit = min(latest["BB_lower"] * 1.005, entry * 0.94)
    else:
        signal = "🤝 HOLD"
        score = max(score_buy, score_sell)
        commentaire = "Aucun signal fort."
        confiance = int((score / 7) * 10)
        confiance_txt = "Faible"
        stop_loss = take_profit = None

    return signal, score, commentaire, stop_loss, take_profit, confiance, confiance_txt

# --- Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await accueil(update, context)

async def accueil(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 Analyse", callback_data="menu_analyse")],
        [InlineKeyboardButton("🏆 Classement", callback_data="menu_classement")],
        [InlineKeyboardButton("ℹ️ Aide", callback_data="menu_help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    chat_id = update.effective_chat.id
    await context.bot.send_message(
        chat_id=chat_id,
        text="👋 Bienvenue sur le bot d'analyse crypto de Luca !",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ Ce bot fournit des signaux d'achat et de vente basés sur des indicateurs techniques.\n"
        "Utilisez les boutons pour interagir.",
        parse_mode=ParseMode.MARKDOWN
    )

async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    if data == "menu_analyse":
        await analyse_callback(update, context)
    elif data == "menu_classement":
        await classement_callback(update, context)
    elif data == "menu_help":
        await help_command(update, context)
    elif data == "retour_accueil":
        await accueil(update, context)

async def analyse_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton(name.title(), callback_data=f"analyse_{symbol}")]
        for name, symbol in TOP_TOKENS
    ]
    keyboard.append([InlineKeyboardButton("⬅️ Retour", callback_data="retour_accueil")])
    await update.callback_query.message.reply_text(
        "📊 Sélectionnez une crypto à analyser :",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

async def analyse_token_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.callback_query.data.replace("analyse_", "")
    name = next((n for n, s in TOP_TOKENS if s == symbol), symbol)
    df = get_binance_ohlc(symbol)
    if df is None or len(df) < 50:
        await update.callback_query.message.reply_text("❌ Données insuffisantes.")
        return
    df = compute_indicators(df)
    signal, score, commentaire, stop_loss, take_profit, confiance, confiance_txt = generate_signal_and_score(df)
    latest = df.iloc[-1]

    # Message détaillé avec indicateurs et icônes
    valid_icons = "✅"
    invalid_icons = "❌"
    indicators = {
        "SMA200": "Moyenne mobile 200 (SMA200)",
        "EMA10": "Moyenne mobile exponentielle 10 (EMA10)",
        "RSI": "RSI (Relative Strength Index)",
        "MACD": "MACD",
        "ADX": "ADX",
        "BB_lower": "Bollinger Bands Lower",
        "BB_upper": "Bollinger Bands Upper",
        "volume_mean": "Volume moyen"
    }

    indicator_status = ""
    for indicator, label in indicators.items():
        status = valid_icons if latest[indicator] else invalid_icons
        indicator_status += f"{status} {label}\n"

    msg = (
        f"*Analyse de {name.title()} ({symbol})*\n"
        f"Prix actuel : `{latest['close']:.2f}` USDT\n"
        f"Signal : {signal}\n"
        f"Score : `{score}/7` | Confiance : `{confiance}/10` ({confiance_txt})\n"
        f"_{commentaire}_\n\n"
        f"*Indicateurs :*\n{indicator_status}\n"
    )
    
    if signal != "🤝 HOLD":
        msg += (
            f"\n🎯 *Take Profit* : `{take_profit:.4f}`\n"
            f"🛑 *Stop Loss* : `{stop_loss:.4f}`"
        )
    
    keyboard = [
        [InlineKeyboardButton("⬅️ Retour", callback_data="retour_accueil")]
    ]
    await update.callback_query.message.reply_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

async def classement_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = await update.callback_query.message.reply_text("🔄 Chargement du classement...")
    results = []
    for name, symbol in TOP_TOKENS:
        df = get_binance_ohlc(symbol)
        if df is None: continue
        df = compute_indicators(df)
        signal, score, commentaire, stop_loss, take_profit, confiance, confiance_txt = generate_signal_and_score(df)
        if signal != "🤝 HOLD":
            results.append((name.title(), signal, score, confiance, commentaire))

    if not results:
        await message.edit_text("Aucun signal fort détecté.")
        return

    results.sort(key=lambda x: (-x[2], -x[3], x[0]))
    msg = "*🏆 Classement des signaux forts :*\n\n"
    for i, (name, signal, score, confiance, commentaire) in enumerate(results, 1):
        msg += f"{i}. *{name}* — {signal} (Score {score}/7, {confiance}/10)\n_{commentaire}_\n\n"
    keyboard = [
        [InlineKeyboardButton("⬅️ Retour", callback_data="retour_accueil")]
    ]
    await message.edit_text(msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Exception:", exc_info=context.error)
    if update and hasattr(update, "effective_chat") and update.effective_chat:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="❌ Une erreur est survenue. Merci de réessayer plus tard."
        )

# --- Lancement bot ---
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(menu_handler, pattern="^menu_|retour_"))
    app.add_handler(CallbackQueryHandler(analyse_token_callback, pattern="^analyse_"))
    app.add_error_handler(error_handler)
    logger.info("Bot lancé et opérationnel.")
    app.run_polling()
