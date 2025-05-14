import os
import logging
import requests
import pandas as pd
import ta
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.helpers import escape_markdown
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

    if latest["close"] > latest["SMA200"]:
        score_buy += 1
    if latest["EMA10"] > latest["SMA20"]:
        score_buy += 1
    if latest["RSI"] < 40:
        score_buy += 1
    if latest["close"] < latest["BB_lower"]:
        score_buy += 1
    if latest["MACD"] > 0:
        score_buy += 1
    if latest["ADX"] > 20:
        score_buy += 1
    if latest["volume"] > 1.2 * latest["volume_mean"]:
        score_buy += 1

    if latest["close"] < latest["SMA200"]:
        score_sell += 1
    if latest["EMA10"] < latest["SMA20"]:
        score_sell += 1
    if latest["RSI"] > 60:
        score_sell += 1
    if latest["close"] > latest["BB_upper"]:
        score_sell += 1
    if latest["MACD"] < 0:
        score_sell += 1
    if latest["ADX"] > 20:
        score_sell += 1
    if latest["volume"] > 1.2 * latest["volume_mean"]:
        score_sell += 1

    if score_buy >= 4 and score_buy >= score_sell:
        signal = "📈 BUY"
        score = score_buy
        commentaire = f"Signal d'achat ({score_buy}/7 critères validés)."
        confiance = int((score_buy / 7) * 10)
        confiance_txt = "Forte" if score_buy >= 6 else "Bonne" if score_buy == 5 else "Moyenne" if score_buy == 4 else "Faible"
        entry = latest["close"]
        sma200 = latest["SMA200"] if not pd.isna(latest["SMA200"]) else entry
        bb_upper = latest["BB_upper"] if not pd.isna(latest["BB_upper"]) else entry
        sl1 = sma200 * 0.997
        sl2 = entry * 0.97
        stop_loss = min(sl1, sl2)
        tp1 = bb_upper * 0.995
        tp2 = entry * 1.06
        take_profit = max(tp1, tp2)
    elif score_sell >= 4 and score_sell > score_buy:
        signal = "📉 SELL"
        score = score_sell
        commentaire = f"Signal de vente ({score_sell}/7 critères validés)."
        confiance = int((score_sell / 7) * 10)
        confiance_txt = "Forte" if score_sell >= 6 else "Bonne" if score_sell == 5 else "Moyenne" if score_sell == 4 else "Faible"
        entry = latest["close"]
        sma200 = latest["SMA200"] if not pd.isna(latest["SMA200"]) else entry
        bb_lower = latest["BB_lower"] if not pd.isna(latest["BB_lower"]) else entry
        sl1 = sma200 * 1.003
        sl2 = entry * 1.03
        stop_loss = max(sl1, sl2)
        tp1 = bb_lower * 1.005
        tp2 = entry * 0.94
        take_profit = min(tp1, tp2)
    else:
        signal = "🤝 HOLD"
        score = max(score_buy, score_sell)
        commentaire = "Aucun signal fort. Tendance neutre ou mitigée."
        confiance = int((score / 7) * 10)
        confiance_txt = "Faible"
        stop_loss = None
        take_profit = None

    return signal, score, commentaire, stop_loss, take_profit, confiance, confiance_txt

# ... (Pas besoin de modifier accueil, start, help_command, menu_handler, help_callback, error_handler)

# Classement avec échappement Markdown
async def classement_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = query.message.chat_id
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    message = await context.bot.send_message(chat_id=chat_id, text="🔎 Analyse des tokens de Luca en cours...")

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
            signal, score, commentaire, stop_loss, take_profit, confiance, confiance_txt = generate_signal_and_score(df)
            latest = df.iloc[-1]

            if signal != "🤝 HOLD":
                results.append({
                    "name": name.title(),
                    "symbol": symbol,
                    "signal": signal,
                    "score": score,
                    "price": latest["close"],
                    "commentaire": commentaire,
                    "confiance": confiance,
                    "confiance_txt": confiance_txt
                })
                progress_msg += (
                    f"✅ {name.title()} : Signal {signal} | Score {score}/7 | "
                    f"Confiance {confiance}/10 ({confiance_txt})\n"
                )
            else:
                progress_msg += f"➖ {name.title()} : Aucun signal fort.\n"

            await context.bot.edit_message_text(progress_msg, chat_id=chat_id, message_id=message.message_id)
            await asyncio.sleep(0.2)
        except Exception as e:
            logger.error(f"Erreur analyse {name}: {e}")
            progress_msg += f"⚠️ {name.title()} : erreur pendant l'analyse.\n"
            await context.bot.edit_message_text(progress_msg, chat_id=chat_id, message_id=message.message_id)

    if not results:
        await context.bot.edit_message_text(
            "Aucun signal fort détecté.", chat_id=chat_id, message_id=message.message_id
        )
        await accueil(update, context)
        return

    results = sorted(results, key=lambda x: (-x["score"], -x["confiance"], x["name"]))

    final_msg = progress_msg + "\n*🏆 Classement des tokens avec signal fort :*\n"
    for i, res in enumerate(results, 1):
        final_msg += (
            f"\n*{escape_markdown(str(i) + '. ' + res['name'], version=2)}* "
            f"(`{escape_markdown(res['symbol'], version=2)}`)\n"
            f"   Prix : `{res['price']:.4f}` USDT\n"
            f"   Signal : {res['signal']} | Score : `{res['score']}/7`\n"
            f"   Confiance : `{res['confiance']}/10` "
            f"({escape_markdown(res['confiance_txt'], version=2)})\n"
            f"   _{escape_markdown(res['commentaire'], version=2)}_\n"
        )
    keyboard = [[InlineKeyboardButton("⬅️ Retour", callback_data="retour_accueil")]]
    await context.bot.edit_message_text(
        final_msg,
        chat_id=chat_id,
        message_id=message.message_id,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN_V2
    )

# ... (Autres fonctions comme analyse_callback et analyse_token_callback doivent aussi utiliser escape_markdown, comme montré plus tôt.)

# --- Lancement du bot ---
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(menu_handler, pattern="^(menu_|retour_|help)"))
    app.add_handler(CallbackQueryHandler(analyse_token_callback, pattern="^analyse_"))
    app.add_error_handler(error_handler)
    logger.info("Bot lancé et en attente de commandes.")
    app.run_polling()
