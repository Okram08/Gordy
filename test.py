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
        take_profit = entry + 2 * atr
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
        take_profit = entry - 2 * atr
    else:
        signal = "🤝 HOLD"
        score = max(score_buy, score_sell)
        commentaire = "Aucun signal fort."
        confiance = int((score / 7) * 10)
        confiance_txt = "Faible"
        stop_loss = take_profit = None

    return signal, score, commentaire, stop_loss, take_profit, confiance, confiance_txt, latest

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

# --- Handlers Telegram (inchangés, sauf backtest_run_callback) ---

async def backtest_run_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = update.callback_query.data.replace("backtest_run_", "")
    symbol, period_code = data.rsplit("_", 1)
    start_date = get_start_date(period_code)
    df = get_binance_ohlc(symbol, interval="1h", limit=1000)
    if df is None or len(df) < 50:
        await update.callback_query.message.reply_text("❌ Données insuffisantes pour le backtest.")
        return

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df[df.index >= start_date]
    if len(df) < 50:
        await update.callback_query.message.reply_text("❌ Pas assez de données pour cette période.")
        return
    df = compute_indicators(df)

    df_8h = df[df.index.hour == 8]
    df_8h = df_8h.groupby(df_8h.index.date).first()

    signals = []
    for idx in df_8h.index:
        dt_8h = pd.Timestamp(idx).replace(hour=8, minute=0, second=0, microsecond=0, tzinfo=df.index.tz)
        if dt_8h < df.index[0]:
            continue
        subdf = df.loc[df.index <= dt_8h]
        if len(subdf) < 50:
            continue
        signal, *_ = generate_signal_and_score(subdf)
        close = subdf.iloc[-1]["close"]
        signals.append({"date": dt_8h, "signal": signal, "close": close})

    trades = []
    current_trade = None
    for sig in signals:
        if current_trade is None:
            if sig["signal"] == "📈 BUY":
                current_trade = {"type": "BUY", "entry_date": sig["date"], "entry_price": sig["close"]}
            elif sig["signal"] == "📉 SELL":
                current_trade = {"type": "SELL", "entry_date": sig["date"], "entry_price": sig["close"]}
        else:
            if (current_trade["type"] == "BUY" and sig["signal"] == "📉 SELL") or \
               (current_trade["type"] == "SELL" and sig["signal"] == "📈 BUY"):
                current_trade["exit_date"] = sig["date"]
                current_trade["exit_price"] = sig["close"]
                if current_trade["type"] == "BUY":
                    current_trade["pnl"] = (current_trade["exit_price"] - current_trade["entry_price"]) / current_trade["entry_price"] * 100
                else:
                    current_trade["pnl"] = (current_trade["entry_price"] - current_trade["exit_price"]) / current_trade["entry_price"] * 100
                trades.append(current_trade)
                if sig["signal"] == "📈 BUY":
                    current_trade = {"type": "BUY", "entry_date": sig["date"], "entry_price": sig["close"]}
                else:
                    current_trade = {"type": "SELL", "entry_date": sig["date"], "entry_price": sig["close"]}
    if current_trade is not None and "exit_price" not in current_trade:
        last = signals[-1]
        current_trade["exit_date"] = last["date"]
        current_trade["exit_price"] = last["close"]
        if current_trade["type"] == "BUY":
            current_trade["pnl"] = (current_trade["exit_price"] - current_trade["entry_price"]) / current_trade["entry_price"] * 100
        else:
            current_trade["pnl"] = (current_trade["entry_price"] - current_trade["exit_price"]) / current_trade["entry_price"] * 100
        trades.append(current_trade)

    nb_trades = len(trades)
    trades_gagnants = [t for t in trades if t["pnl"] > 0]
    trades_perdants = [t for t in trades if t["pnl"] <= 0]
    taux_reussite = (len(trades_gagnants) / nb_trades) * 100 if nb_trades else 0
    pnl_total = sum(t["pnl"] for t in trades)
    pnl_moyen = pnl_total / nb_trades if nb_trades else 0
    max_drawdown = 0
    equity = 0
    peak = 0
    equity_curve = []
    for t in trades:
        equity += t["pnl"]
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    msg = (
        f"📊 *Backtest {symbol} sur {period_code}*\n"
        f"╭─────────────────────────────╮\n"
        f"│ Jours analysés : *{len(df_8h)}*\n"
        f"│ Trades simulés : *{nb_trades}*\n"
        f"│ 🟩 Trades gagnants : *{len(trades_gagnants)}*\n"
        f"│ 🟥 Trades perdants : *{len(trades_perdants)}*\n"
        f"│ 📈 Taux de réussite : *{taux_reussite:.1f}%*\n"
        f"│ 💰 P&L total : *{pnl_total:.2f}%*\n"
        f"│ ⚖️ P&L moyen/trade : *{pnl_moyen:.2f}%*\n"
        f"│ 📉 Max drawdown : *{max_drawdown:.2f}%*\n"
        f"╰─────────────────────────────╯\n"
        f"_Entrée/Sortie sur chaque changement de signal à 8h UTC._"
    )
    keyboard = [
        [InlineKeyboardButton("⬅️ Retour", callback_data=f"backtest_{symbol}")],
        [InlineKeyboardButton("🏠 Accueil", callback_data="retour_accueil")]
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
        signal, score, commentaire, stop_loss, take_profit, confiance, confiance_txt, latest = generate_signal_and_score(df)
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
    app.add_handler(CallbackQueryHandler(backtest_menu_callback, pattern="^backtest_((?!run).)+$"))
    app.add_handler(CallbackQueryHandler(backtest_run_callback, pattern="^backtest_run_"))
    app.add_error_handler(error_handler)
    logger.info("Bot lancé et opérationnel.")
    app.run_polling()
