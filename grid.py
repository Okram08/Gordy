import os
import time
import ccxt
import numpy as np
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler

# Charger les variables d'environnement
load_dotenv()

# Clés d'API
api_key = os.getenv('HYPERLIQUID_API_KEY')
api_secret = os.getenv('HYPERLIQUID_API_SECRET')
private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
telegram_api_key = os.getenv('TELEGRAM_API_KEY')

# Initialiser la connexion à Hyperliquid (ou autre API d'échange)
exchange = ccxt.hyperliquid({
    'apiKey': api_key,
    'secret': api_secret,
    'privateKey': private_key
})

# Fonction pour calculer le RSI (Relative Strength Index) manuellement
def calculate_rsi(prices, period=14):
    gains = []
    losses = []

    # Calcul des gains et pertes
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            losses.append(-change)
            gains.append(0)

    # Calcul des moyennes des gains et pertes
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100  # Eviter la division par zéro si aucune perte

    # Calcul du RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fonction d'analyse technique (RSI)
def analyse_token(prices):
    rsi = calculate_rsi(prices)
    if rsi > 70:
        return "suracheté"
    elif rsi < 30:
        return "sous-évalué"
    else:
        return "neutre"

# Fonction pour le Grid Trading
def grid_trading(pair, buy_price, sell_price, grid_size):
    for i in range(grid_size):
        buy_order = exchange.create_limit_buy_order(pair, 1, buy_price - (i * 0.01))
        sell_order = exchange.create_limit_sell_order(pair, 1, sell_price + (i * 0.01))
        print(f'Ordre d\'achat: {buy_order}')
        print(f'Ordre de vente: {sell_order}')

# Fonction pour démarrer le bot Telegram
def start(update: Update, context):
    update.message.reply_text('Salut, je suis ton bot de trading Grid!')

def stop(update: Update, context):
    update.message.reply_text('Je m\'arrête, à bientôt!')

# Lancer le bot Telegram
def main():
    # Création de l'instance de l'Application et du Bot
    application = Application.builder().token(telegram_api_key).build()

    # Ajout des handlers pour les commandes
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))

    # Lancer le bot
    application.run_polling()

# Fonction pour vérifier et mettre à jour les tokens disponibles à intervalles réguliers
def check_and_update_tokens():
    while True:
        # Récupérer les tokens disponibles
        markets = exchange.load_markets()
        for token in markets:
            try:
                prices = exchange.fetch_ohlcv(token, timeframe='1h')  # Récupérer les prix sur 1h
                prices = [price[4] for price in prices]  # Extraire les prix de clôture
                status = analyse_token(prices)
                if status == "sous-évalué":
                    # Commencer à trader sur ce token
                    grid_trading(token, prices[-1] * 0.98, prices[-1] * 1.02, 5)
            except Exception as e:
                print(f"Erreur lors de l'analyse du token {token}: {e}")

        # Attendre 10 minutes avant de vérifier à nouveau
        time.sleep(600)

# Lancer le bot et la mise à jour des tokens
if __name__ == '__main__':
    # Lancer le bot Telegram dans un thread séparé
    from threading import Thread
    telegram_thread = Thread(target=main)
    telegram_thread.start()

    # Lancer la vérification et mise à jour des tokens
    check_and_update_tokens()
