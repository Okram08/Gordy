import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes,
    ConversationHandler, filters
)
from pycoingecko import CoinGeckoAPI

# --- Logging configuration: affichage en temps réel dans le terminal ---
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

def get_price_or_token_price(user_input):
    user_input = user_input.strip().lower()
    # Si c'est une adresse de contrat ERC20 Ethereum
    if user_input.startswith("0x") and len(user_input) == 42:
        try:
            result = cg.get_token_price(
                id="ethereum",
                contract_addresses=user_input,
                vs_currencies='usd'
            )
            logger.info(f"Réponse CoinGecko pour contrat {user_input}: {result}")
            price = result.get(user_input, {}).get('usd')
            return price
        except Exception as e:
            logger.error(f"Erreur CoinGecko ERC20: {e}")
            return None
    else:
        # Sinon, on suppose que c'est un id CoinGecko (crypto native)
        try:
            result = cg.get_price(ids=user_input, vs_currencies='usd')
            logger.info(f"Réponse CoinGecko pour natif {user_input}: {result}")
            price = result.get(user_input, {}).get('usd')
            return price
        except Exception as e:
            logger.error(f"Erreur CoinGecko natif: {e}")
            return None

# --- Handlers Telegram ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} started the bot.")
    await update.message.reply_text(
        "Bienvenue ! Envoie-moi le nom d'une crypto (ex: bitcoin, ethereum) ou l'adresse du token ERC20 (ex: 0x...) que tu veux analyser."
    )
    return ASK_TOKEN

async def ask_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    logger.info(f"User {update.effective_user.id} demande le token: {user_input}")
    price = get_price_or_token_price(user_input)
    if price:
        await update.message.reply_text(
            f"Le prix actuel de '{user_input}' est : {price:.4f} USD"
        )
        logger.info(f"Réponse envoyée pour {user_input}")
    else:
        await update.message.reply_text(
            "Impossible de récupérer le prix pour cette entrée. "
            "Envoie un nom valide (ex: bitcoin, ethereum) ou une adresse de contrat ERC20 (ex: 0x...)."
        )
        logger.warning(f"Echec récupération prix pour {user_input}")
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
