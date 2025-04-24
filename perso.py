import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes,
    ConversationHandler, filters
)
from moralis import evm_api

# --- Logging configuration: affichage en temps réel dans le terminal ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

# --- Chargement des variables d'environnement ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MORALIS_API_KEY = os.getenv("MORALIS_API_KEY")

# --- États de la conversation ---
ASK_TOKEN = 1

# --- Fonction pour récupérer le prix du token via Moralis ---
def get_token_price(token_address, chain="eth"):
    try:
        params = {"address": token_address, "chain": chain}
        result = evm_api.token.get_token_price(api_key=MORALIS_API_KEY, params=params)
        logger.info(f"Prix récupéré pour {token_address}: {result['usdPrice']} USD")
        return result["usdPrice"]
    except Exception as e:
        logger.error(f"Erreur Moralis API: {e}")
        return None

# --- Handlers Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} started the bot.")
    await update.message.reply_text(
        "Bienvenue ! Envoie-moi l'adresse du token (contract address ERC20) que tu veux analyser."
    )
    return ASK_TOKEN

async def ask_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    token_address = update.message.text.strip()
    logger.info(f"User {update.effective_user.id} demande le token: {token_address}")
    price = get_token_price(token_address)
    if price:
        await update.message.reply_text(
            f"Le prix actuel du token ({token_address}) est : {price:.4f} USD"
        )
        logger.info(f"Réponse envoyée pour le token {token_address}")
    else:
        await update.message.reply_text(
            "Impossible de récupérer le prix pour ce token. Vérifie l'adresse et réessaie."
        )
        logger.warning(f"Echec récupération prix pour {token_address}")
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
