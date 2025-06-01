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

# --- Backtest avec trailing stop et prise de profit partielle ---
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
            "take_profit": take_profit,
            "atr": latest["ATR"]
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
        atr = sig["atr"]

        # Prise de profit partielle à +1 ATR, trailing stop à +0.5 ATR après prise partielle
        partial_tp = entry_price + atr if trade_type == "BUY" else entry_price - atr
        trailing_active = False
        trailing_stop = None
        partial_taken = False

        size_full = 1.0
        size_left = 1.0
        pnl_partial = 0
        pnl_final = 0

        df_after = df[df.index > entry_date]
        exit_reason = None
        exit_date = None
        exit_price = None

        for idx, row in df_after.iterrows():
            # Prise de profit partielle
            if not partial_taken:
                if (trade_type == "BUY" and row["high"] >= partial_tp) or (trade_type == "SELL" and row["low"] <= partial_tp):
                    # Prise de profit sur la moitié
                    partial_taken = True
                    size_left = 0.5
                    pnl_partial = ((partial_tp - entry_price) / entry_price * 100) * 0.5 if trade_type == "BUY" else ((entry_price - partial_tp) / entry_price * 100) * 0.5
                    # Active le trailing stop sur le reste
                    trailing_active = True
                    if trade_type == "BUY":
                        trailing_stop = partial_tp - 0.5 * atr
                    else:
                        trailing_stop = partial_tp + 0.5 * atr
                    # Continue pour la partie restante
            # TP/SL classiques
            if (trade_type == "BUY" and row["low"] <= stop_loss):
                pnl_final = ((stop_loss - entry_price) / entry_price * 100) * size_left
                exit_reason = "SL"
                exit_date = idx
                exit_price = stop_loss
                break
            if (trade_type == "SELL" and row["high"] >= stop_loss):
                pnl_final = ((entry_price - stop_loss) / entry_price * 100) * size_left
                exit_reason = "SL"
                exit_date = idx
                exit_price = stop_loss
                break
            if (trade_type == "BUY" and row["high"] >= take_profit):
                pnl_final = ((take_profit - entry_price) / entry_price * 100) * size_left
                exit_reason = "TP"
                exit_date = idx
                exit_price = take_profit
                break
            if (trade_type == "SELL" and row["low"] <= take_profit):
                pnl_final = ((entry_price - take_profit) / entry_price * 100) * size_left
                exit_reason = "TP"
                exit_date = idx
                exit_price = take_profit
                break
            # Trailing stop sur la moitié restante
            if trailing_active:
                if trade_type == "BUY":
                    if row["high"] > trailing_stop + 0.5 * atr:
                        trailing_stop = row["high"] - 0.5 * atr
                    if row["low"] <= trailing_stop:
                        pnl_final = ((trailing_stop - entry_price) / entry_price * 100) * size_left
                        exit_reason = "Trailing Stop"
                        exit_date = idx
                        exit_price = trailing_stop
                        break
                else:
                    if row["low"] < trailing_stop - 0.5 * atr:
                        trailing_stop = row["low"] + 0.5 * atr
                    if row["high"] >= trailing_stop:
                        pnl_final = ((entry_price - trailing_stop) / entry_price * 100) * size_left
                        exit_reason = "Trailing Stop"
                        exit_date = idx
                        exit_price = trailing_stop
                        break
            # Signal opposé à 8h
            if idx.hour == 8 and idx.date() != entry_date.date():
                opp = "📉 SELL" if trade_type == "BUY" else "📈 BUY"
                next_sig = next((s for s in signals if s["date"] == idx and s["signal"] == opp), None)
                if next_sig:
                    if trade_type == "BUY":
                        pnl_final = ((row["open"] - entry_price) / entry_price * 100) * size_left
                    else:
                        pnl_final = ((entry_price - row["open"]) / entry_price * 100) * size_left
                    exit_reason = "Signal Opposé"
                    exit_date = idx
                    exit_price = row["open"]
                    break

        # Si pas de sortie, on sort à la dernière bougie
        if exit_reason is None:
            last_idx = df_after.index[-1] if not df_after.empty else df.index[-1]
            if trade_type == "BUY":
                pnl_final = ((df.loc[last_idx]["close"] - entry_price) / entry_price * 100) * size_left
            else:
                pnl_final = ((entry_price - df.loc[last_idx]["close"]) / entry_price * 100) * size_left
            exit_reason = "Fin période"
            exit_date = last_idx
            exit_price = df.loc[exit_date]["close"]

        pnl = pnl_partial + pnl_final

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
        f"_Prise de profit partielle à +1 ATR, trailing stop sur le reste._"
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
