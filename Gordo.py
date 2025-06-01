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
from datetime import datetime, timezone, timedelta

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# --- Env ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

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

def get_binance_ohlc(symbol, interval="1h", limit=1000):
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
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
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
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    for col in ["SMA20", "EMA10", "RSI", "SMA200", "MACD", "ADX", "volume_mean", "BB_upper", "BB_lower", "ATR"]:
        df[col] = df[col].ffill().bfill().fillna(df["close"])
    return df

def generate_signal_and_score(df):
    latest = df.iloc[-1]
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

    atr = latest["ATR"]
    entry = latest["close"]

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
        recent_lows = df["low"].iloc[-20:]
        stop_loss = min(recent_lows.min(), entry - 1.5 * atr)
        tp1 = entry + atr
        tp2 = "Trailing stop à 0.5 ATR sous le plus haut atteint"
        take_profit = entry + 2 * atr  # pour compatibilité backtest
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
        recent_highs = df["high"].iloc[-20:]
        stop_loss = max(recent_highs.max(), entry + 1.5 * atr)
        tp1 = entry - atr
        tp2 = "Trailing stop à 0.5 ATR au-dessus du plus bas atteint"
        take_profit = entry - 2 * atr  # pour compatibilité backtest
    else:
        signal = "🤝 HOLD"
        score = max(score_buy, score_sell)
        commentaire = "Aucun signal fort."
        confiance = int((score / 7) * 10)
        confiance_txt = "Faible"
        stop_loss = take_profit = tp1 = tp2 = None

    return signal, score, commentaire, stop_loss, take_profit, confiance, confiance_txt, latest, tp1, tp2

def get_criteria_status(latest, signal_type):
    if signal_type == "BUY":
        criteria = [
            ("Prix > SMA200", latest["close"] > latest["SMA200"]),
            ("EMA10 > SMA20", latest["EMA10"] > latest["SMA20"]),
            ("RSI < 40", latest["RSI"] < 40),
            ("Prix < Bande Basse Bollinger", latest["close"] < latest["BB_lower"]),
            ("MACD > 0", latest["MACD"] > 0),
            ("ADX > 20", latest["ADX"] > 20),
            ("Volume > 1.2x volume moyen", latest["volume"] > 1.2 * latest["volume_mean"]),
        ]
    elif signal_type == "SELL":
        criteria = [
            ("Prix < SMA200", latest["close"] < latest["SMA200"]),
            ("EMA10 < SMA20", latest["EMA10"] < latest["SMA20"]),
            ("RSI > 60", latest["RSI"] > 60),
            ("Prix > Bande Haute Bollinger", latest["close"] > latest["BB_upper"]),
            ("MACD < 0", latest["MACD"] < 0),
            ("ADX > 20", latest["ADX"] > 20),
            ("Volume > 1.2x volume moyen", latest["volume"] > 1.2 * latest["volume_mean"]),
        ]
    else:
        criteria = [
            ("Prix > SMA200", latest["close"] > latest["SMA200"]),
            ("EMA10 > SMA20", latest["EMA10"] > latest["SMA20"]),
            ("RSI < 40", latest["RSI"] < 40),
            ("Prix < Bande Basse Bollinger", latest["close"] < latest["BB_lower"]),
            ("MACD > 0", latest["MACD"] > 0),
            ("ADX > 20", latest["ADX"] > 20),
            ("Volume > 1.2x volume moyen", latest["volume"] > 1.2 * latest["volume_mean"]),
            ("Prix < SMA200", latest["close"] < latest["SMA200"]),
            ("EMA10 < SMA20", latest["EMA10"] < latest["SMA20"]),
            ("RSI > 60", latest["RSI"] > 60),
            ("Prix > Bande Haute Bollinger", latest["close"] > latest["BB_upper"]),
            ("MACD < 0", latest["MACD"] < 0),
        ]
    return criteria

def get_start_date(period_code):
    now = datetime.now(timezone.utc)
    if period_code == "1m":
        return now - timedelta(days=30)
    elif period_code == "3m":
        return now - timedelta(days=90)
    elif period_code == "6m":
        return now - timedelta(days=180)
    elif period_code == "1y":
        return now - timedelta(days=365)
    else:
        return now - timedelta(days=30)

# --- Handlers ---
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
    signal, score, commentaire, stop_loss, take_profit, confiance, confiance_txt, latest, tp1, tp2 = generate_signal_and_score(df)

    if signal == "📈 BUY":
        criteria = get_criteria_status(latest, "BUY")
    elif signal == "📉 SELL":
        criteria = get_criteria_status(latest, "SELL")
    else:
        criteria = get_criteria_status(latest, "HOLD")

    indicator_status = ""
    for label, valid in criteria:
        icon = "✅" if valid else "❌"
        indicator_status += f"{icon} {label}\n"

    msg = (
        f"*Analyse de {name.title()} ({symbol})*\n"
        f"Prix actuel : `{latest['close']:.2f}` USDT\n"
        f"Signal : {signal}\n"
        f"Score : `{score}/7` | Confiance : `{confiance}/10` ({confiance_txt})\n"
        f"_{commentaire}_\n\n"
        f"*Critères validés :*\n{indicator_status}\n"
    )

    if signal != "🤝 HOLD":
        msg += (
            f"\n🎯 *Take Profit 1* (50%) : `{tp1:.4f}`\n"
            f"🎯 *Take Profit 2* (reste) : {tp2}\n"
            f"🛑 *Stop Loss* : `{stop_loss:.4f}`"
        )

    keyboard = [
        [InlineKeyboardButton("Backtest 🔄", callback_data=f"backtest_{symbol}")],
        [InlineKeyboardButton("⬅️ Retour", callback_data="retour_accueil")]
    ]
    await update.callback_query.message.reply_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

# (Le reste du code, y compris le backtest, le classement, l'error handler, et le lancement du bot, reste inchangé.)

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(menu_handler, pattern="^menu_|retour_"))
    app.add_handler(CallbackQueryHandler(analyse_token_callback, pattern="^analyse_"))
    app.add_handler(CallbackQueryHandler(backtest_menu_callback, pattern="^backtest_((?!run).)+$"))
    app.add_handler(CallbackQueryHandler(backtest_run_callback, pattern="^backtest_run_"))
    app.add_error_handler(error_handler)
    logger.info("Bot lancé et opérationnel.")
    app.run_polling()
