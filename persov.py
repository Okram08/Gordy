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

# --- Ecran d'accueil ---
async def accueil(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🏆 Classement", callback_data="menu_classement")],
        [InlineKeyboardButton("📊 Analyse", callback_data="menu_analyse")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    chat_id = update.effective_chat.id if update.effective_chat else update.callback_query.message.chat_id
    await context.bot.send_message(
        chat_id=chat_id,
        text="👋 Bienvenue ! Que souhaitez-vous faire ?",
        reply_markup=reply_markup
    )

# --- Gestion des menus ---
async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Répondre au CallbackQuery pour éviter un timeout
    
    # Gérer les différentes options du menu
    if query.data == "menu_classement":
        await query.edit_message_text(text="Classement des tokens en cours...")
        
        classement = "\n".join([f"{idx+1}. {name.title()} ({symbol})" for idx, (name, symbol) in enumerate(TOP_TOKENS)])
        await query.edit_message_text(text=f"Voici le classement des tokens :\n{classement}")

    elif query.data == "menu_analyse":
        await query.edit_message_text(text="Sélectionne un token à analyser.")
        
        # Créer un clavier dynamique avec les crypto-monnaies
        keyboard = [
            [InlineKeyboardButton(name.title(), callback_data=f"analyse_{symbol}")]
            for name, symbol in TOP_TOKENS
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text="Cliquez sur un token pour analyser", reply_markup=reply_markup)

    else:
        await query.edit_message_text(text="Option non reconnue.")

# --- Analyse d'un token ---
async def analyse_token_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Répondre au CallbackQuery pour éviter un timeout

    # Récupérer le symbole du token
    symbol = query.data.replace("analyse_", "")
    name = next((name.title() for name, sym in TOP_TOKENS if sym == symbol), symbol)
    
    await query.edit_message_text(text=f"🔍 Analyse de {name} ({symbol}) en cours...")

    # Récupérer les données pour le token
    df = get_binance_ohlc(symbol)
    if df is None or len(df) < 50:
        await query.edit_message_text(text=f"❌ Pas assez de données pour {name}.")
        await accueil(update, context)
        return

    # Calculer les indicateurs
    df = compute_indicators(df)
    latest = df.iloc[-1]
    result = (
        f"📊 Résultat pour {name} ({symbol}):\n"
        f"Prix : {latest['close']:.4f} USDT\n"
    )
    await query.edit_message_text(text=result)
    await accueil(update, context)

# --- Commande /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await accueil(update, context)

# --- Main ---
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(menu_handler, pattern="^menu_"))
    app.add_handler(CallbackQueryHandler(analyse_token_callback, pattern="^analyse_"))
    logger.info("Bot lancé et en attente de commandes.")
    app.run_polling()
