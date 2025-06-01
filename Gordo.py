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

# --- Paramètres stratégie ---
ATR_MIN = 0.5  # à ajuster selon la volatilité de tes actifs
TRAILING_ATR = 1.0  # trailing stop à 1 ATR

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
    df["SMA200_prev"] = df["SMA200"].shift(24)
    df["MACD"] = ta.trend.macd_diff(df["close"])
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["volume_mean"] = df["volume"].rolling(window=20, min_periods=1).mean()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    for col in ["SMA20", "EMA10", "RSI", "SMA200", "SMA200_prev", "MACD", "ADX", "volume_mean", "BB_upper", "BB_lower", "ATR"]:
        df[col] = df[col].ffill().bfill().fillna(df["close"])
    return df

def generate_signal_and_score(df):
    latest = df.iloc[-1]
    # Filtre tendance et volatilité
    trend_up = latest["close"] > latest["SMA200"] and latest["SMA200"] > latest["SMA200_prev"]
    trend_down = latest["close"] < latest["SMA200"] and latest["SMA200"] < latest["SMA200_prev"]
    atr_ok = latest["ATR"] > ATR_MIN

    # BUY si toutes les conditions majeures sont réunies
    buy_ok = (
        trend_up and
        latest["EMA10"] > latest["SMA20"] and
        latest["RSI"] < 40 and
        latest["MACD"] > 0 and
        latest["ADX"] > 20 and
        atr_ok
    )
    # SELL si toutes les conditions majeures sont réunies
    sell_ok = (
        trend_down and
        latest["EMA10"] < latest["SMA20"] and
        latest["RSI"] > 60 and
        latest["MACD"] < 0 and
        latest["ADX"] > 20 and
        atr_ok
    )

    atr = latest["ATR"]
    entry = latest["close"]
    recent_lows = df["low"].iloc[-20:]
    recent_highs = df["high"].iloc[-20:]

    if buy_ok:
        signal = "📈 BUY"
        commentaire = "Signal d'achat optimal (tendance, momentum, volatilité OK)."
        confiance = 10
        confiance_txt = "Forte"
        stop_loss = min(recent_lows.min(), entry - 1.5 * atr)
        take_profit = entry + 2 * atr
    elif sell_ok:
        signal = "📉 SELL"
        commentaire = "Signal de vente optimal (tendance, momentum, volatilité OK)."
        confiance = 10
        confiance_txt = "Forte"
        stop_loss = max(recent_highs.max(), entry + 1.5 * atr)
        take_profit = entry - 2 * atr
    else:
        signal = "🤝 HOLD"
        commentaire = "Aucun signal optimal (filtre tendance/momentum/volatilité non validé)."
        confiance = 0
        confiance_txt = "Faible"
        stop_loss = take_profit = None

    # Pour affichage
    score = int(confiance / 2)
    return signal, score, commentaire, stop_loss, take_profit, confiance, confiance_txt, latest

def get_criteria_status(latest, signal_type):
    trend_up = latest["close"] > latest["SMA200"] and latest["SMA200"] > latest["SMA200_prev"]
    trend_down = latest["close"] < latest["SMA200"] and latest["SMA200"] < latest["SMA200_prev"]
    atr_ok = latest["ATR"] > ATR_MIN
    if signal_type == "BUY":
        criteria = [
            ("Tendance haussière (prix > SMA200 et SMA200 monte)", trend_up),
            ("EMA10 > SMA20", latest["EMA10"] > latest["SMA20"]),
            ("RSI < 40", latest["RSI"] < 40),
            ("MACD > 0", latest["MACD"] > 0),
            ("ADX > 20", latest["ADX"] > 20),
            ("Volatilité (ATR) suffisante", atr_ok),
        ]
    elif signal_type == "SELL":
        criteria = [
            ("Tendance baissière (prix < SMA200 et SMA200 baisse)", trend_down),
            ("EMA10 < SMA20", latest["EMA10"] < latest["SMA20"]),
            ("RSI > 60", latest["RSI"] > 60),
            ("MACD < 0", latest["MACD"] < 0),
            ("ADX > 20", latest["ADX"] > 20),
            ("Volatilité (ATR) suffisante", atr_ok),
        ]
    else:
        criteria = []
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
    signal, score, commentaire, stop_loss, take_profit, confiance, confiance_txt, latest = generate_signal_and_score(df)

    if signal == "📈 BUY":
        criteria = get_criteria_status(latest, "BUY")
    elif signal == "📉 SELL":
        criteria = get_criteria_status(latest, "SELL")
    else:
        criteria = []

    indicator_status = ""
    for label, valid in criteria:
        icon = "✅" if valid else "❌"
        indicator_status += f"{icon} {label}\n"

    msg = (
        f"*Analyse de {name.title()} ({symbol})*\n"
        f"Prix actuel : `{latest['close']:.2f}` USDT\n"
        f"Signal : {signal}\n"
        f"Confiance : `{confiance}/10` ({confiance_txt})\n"
        f"_{commentaire}_\n\n"
        f"*Critères validés :*\n{indicator_status}\n"
    )

    if signal != "🤝 HOLD":
        msg += (
            f"\n🎯 *Take Profit* : `{take_profit:.4f}`\n"
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

async def backtest_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.callback_query.data.replace("backtest_", "")
    keyboard = [
        [InlineKeyboardButton("1 mois", callback_data=f"backtest_run_{symbol}_1m")],
        [InlineKeyboardButton("3 mois", callback_data=f"backtest_run_{symbol}_3m")],
        [InlineKeyboardButton("6 mois", callback_data=f"backtest_run_{symbol}_6m")],
        [InlineKeyboardButton("1 an", callback_data=f"backtest_run_{symbol}_1y")],
        [InlineKeyboardButton("⬅️ Retour", callback_data=f"analyse_{symbol}")]
    ]
    await update.callback_query.message.reply_text(
        "🕒 Choisis la période de backtest :",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )

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

    # Génère les signaux à 8h (date, type, prix, TP, SL)
    signals = []
    df_8h = df[df.index.hour == 8]
    df_8h = df_8h.groupby(df_8h.index.date).first()
    for idx in df_8h.index:
        dt_8h = pd.Timestamp(idx).replace(hour=8, minute=0, second=0, microsecond=0, tzinfo=df.index.tz)
        if dt_8h < df.index[0]:
            continue
        subdf = df.loc[df.index <= dt_8h]
        if len(subdf) < 50:
            continue
        signal, _, _, stop_loss, take_profit, _, _, latest = generate_signal_and_score(subdf)
        close = latest["close"]
        signals.append({
            "date": dt_8h,
            "signal": signal,
            "close": close,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        })

    trades = []
    i = 0
    while i < len(signals):
        sig = signals[i]
        if sig["signal"] not in ["📈 BUY", "📉 SELL"]:
            i += 1
            continue

        trade_type = "BUY" if sig["signal"] == "📈 BUY" else "SELL"
        entry_date = sig["date"]
        entry_price = sig["close"]
        stop_loss = sig["stop_loss"]
        take_profit = sig["take_profit"]

        # Trailing stop dynamique
        trailing_stop = entry_price - TRAILING_ATR * df.loc[entry_date]["ATR"] if trade_type == "BUY" else entry_price + TRAILING_ATR * df.loc[entry_date]["ATR"]
        highest = entry_price
        lowest = entry_price

        df_after = df[df.index > entry_date]
        exit_reason = None
        exit_date = None
        exit_price = None

        for idx, row in df_after.iterrows():
            # Trailing stop update
            if trade_type == "BUY":
                if row["high"] > highest:
                    highest = row["high"]
                    trailing_stop = max(trailing_stop, highest - TRAILING_ATR * row["ATR"])
                if row["low"] <= stop_loss:
                    exit_reason = "SL"
                    exit_date = idx
                    exit_price = stop_loss
                    break
                if row["high"] >= take_profit:
                    exit_reason = "TP"
                    exit_date = idx
                    exit_price = take_profit
                    break
                if row["low"] <= trailing_stop:
                    exit_reason = "Trailing Stop"
                    exit_date = idx
                    exit_price = trailing_stop
                    break
            else:
                if row["low"] < lowest:
                    lowest = row["low"]
                    trailing_stop = min(trailing_stop, lowest + TRAILING_ATR * row["ATR"])
                if row["high"] >= stop_loss:
                    exit_reason = "SL"
                    exit_date = idx
                    exit_price = stop_loss
                    break
                if row["low"] <= take_profit:
                    exit_reason = "TP"
                    exit_date = idx
                    exit_price = take_profit
                    break
                if row["high"] >= trailing_stop:
                    exit_reason = "Trailing Stop"
                    exit_date = idx
                    exit_price = trailing_stop
                    break
            # Signal opposé à 8h
            if idx.hour == 8 and idx.date() != entry_date.date():
                opp = "📉 SELL" if trade_type == "BUY" else "📈 BUY"
                next_sig = next((s for s in signals if s["date"] == idx and s["signal"] == opp), None)
                if next_sig:
                    exit_reason = "Signal Opposé"
                    exit_date = idx
                    exit_price = row["open"]
                    break

        if exit_reason is None:
            last_idx = df_after.index[-1] if not df_after.empty else df.index[-1]
            exit_reason = "Fin période"
            exit_date = last_idx
            exit_price = df.loc[exit_date]["close"]

        if trade_type == "BUY":
            pnl = (exit_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - exit_price) / entry_price * 100

        trades.append({
            "type": trade_type,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "reason": exit_reason,
            "pnl": pnl
        })

        next_signals = [j for j, s in enumerate(signals) if s["date"] > exit_date]
        i = next_signals[0] if next_signals else len(signals)

    nb_trades = len(trades)
    trades_gagnants = [t for t in trades if t["pnl"] > 0]
    trades_perdants = [t for t in trades if t["pnl"] <= 0]
    taux_reussite = (len(trades_gagnants) / nb_trades) * 100 if nb_trades else 0
    pnl_total = sum(t["pnl"] for t in trades)
    pnl_moyen = pnl_total / nb_trades if nb_trades else 0
    max_drawdown = 0
    equity = 0
    peak = 0
    for t in trades:
        equity += t["pnl"]
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
        f"_Sortie sur TP, SL, trailing stop ou signal opposé à 8h UTC._"
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
