import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import ccxt
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Obtenir les variables API
telegram_api_key = os.getenv("TELEGRAM_API_KEY")
hyperliquid_api_key = os.getenv("HYPERLIQUID_API_KEY")
hyperliquid_secret = os.getenv("HYPERLIQUID_SECRET")

# Fonction démarrer le bot Telegram
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Salut, je suis ton bot de trading Grid!')

# Fonction pour arrêter le bot Telegram
async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Je m\'arrête, à bientôt!')

# Lancer le bot Telegram
def main():
    # Créer l'instance du bot avec la clé API
    application = Application.builder().token(telegram_api_key).build()

    # Ajouter des handlers pour les commandes
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))

    # Lancer le bot
    application.run_polling()

# Initialiser l'event loop explicitement
if __name__ == '__main__':
    asyncio.set_event_loop(asyncio.new_event_loop())  # Définir l'event loop
    main()

# Exemple de vérification de la connexion Hyperliquid
def check_hyperliquid_api():
    try:
        exchange = ccxt.hyperliquid({
            'apiKey': hyperliquid_api_key,
            'secret': hyperliquid_secret
        })
        tokens = exchange.fetch_markets()
        print("Connexion réussie aux marchés Hyperliquid")
    except Exception as e:
        print(f"Erreur lors de la connexion à l'API Hyperliquid: {e}")

# Vérifier la connexion à l'API Hyperliquid
check_hyperliquid_api()
