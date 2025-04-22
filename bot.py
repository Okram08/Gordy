import logging
import os
import numpy as np
import pandas as pd
from io import BytesIO
from functools import lru_cache
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler
)
from pycoingecko import CoinGeckoAPI
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import json

ASK_TOKEN = 0
CONFIRM_CONTINUE = 1  # Nouvelle étape pour demander à l'utilisateur s'il souhaite continuer ou non
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

cg = CoinGeckoAPI()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

LOOKBACK = 24
TRAIN_TEST_RATIO = 0.8
CLASS_THRESHOLD = 0.003
HISTORY_FILE = 'analysis_history.json'

# ... (le reste de vos fonctions)

async def analyze_and_reply(update: Update, token: str):
    await update.message.reply_text(f"📈 Analyse de {token} en cours...")
    try:
        ohlc = get_crypto_data(token, 30)
        if not ohlc:
            await update.message.reply_text(f"❌ Token {token} non trouvé ou erreur API.")
            return

        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        df['macd'], df['signal'] = compute_macd(df['close'])
        df['rsi'] = compute_rsi(df['close'])
        df['atr'] = compute_atr(df['high'], df['low'], df['close'])
        df = generate_labels(df)

        features = ['rsi', 'macd', 'atr']
        X_train, X_test, y_train, y_test = prepare_data(df, features)

        model_path = os.path.join(MODELS_DIR, f'{token}_clf_model.keras')

        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([
                Input(shape=(X_train.shape[1], X_train.shape[2])),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.2),
                Dense(3, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            model.save(model_path)

        last_sequence = X_test[-1:]
        prediction = model.predict(last_sequence, verbose=0)[0]
        pred_class = np.argmax(prediction)
        confidence = prediction[pred_class]

        direction = "⬆️ LONG" if pred_class == 2 else ("⬇️ SHORT" if pred_class == 0 else "🔁 NEUTRE")

        current_price = get_live_price(token)
        if current_price is None:
            await update.message.reply_text(f"❌ Impossible de récupérer le prix en direct pour {token}. Réessaie plus tard.")
            return

        atr = df['atr'].iloc[-1]
        tp = current_price + 2 * atr if pred_class == 2 else (current_price - 2 * atr if pred_class == 0 else current_price)
        sl = current_price - atr if pred_class == 2 else (current_price + atr if pred_class == 0 else current_price)

        message = (
            f"📊 {token.upper()} - Signal IA\n"
            f"🎯 Direction: {direction}\n"
            f"📈 Confiance: {confidence*100:.2f}%\n"
            f"💰 Prix live: {current_price:.2f}$\n"
            f"🎯 TP: {tp:.2f}$ | 🛑 SL: {sl:.2f}$\n"
        )

        history = load_history()
        result = {
            'token': token,
            'timestamp': str(datetime.now()),
            'direction': direction,
            'confidence': confidence,
            'pred_class': int(pred_class),
            'current_price': float(current_price),
            'tp': float(tp),
            'sl': float(sl)
        }
        history.append(result)
        save_history(history)

        await update.message.reply_text(message)

        # Demander à l'utilisateur s'il veut analyser une autre crypto
        keyboard = [
            [
                {'text': 'Oui, analyser une autre crypto', 'callback_data': 'yes'},
                {'text': 'Non, arrêter', 'callback_data': 'no'}
            ]
        ]
        reply_markup = {'inline_keyboard': keyboard}
        await update.message.reply_text(
            "🔁 Veux-tu analyser une autre cryptomonnaie ?",
            reply_markup=reply_markup
        )

    except Exception as e:
        logging.error(f"Erreur: {str(e)}")
        await update.message.reply_text(f"❌ Une erreur est survenue durant l'analyse.\n🛠 Détail: {str(e)}")


async def handle_continue(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == 'yes':
        await update.message.reply_text("👋 Quel(s) token(s) veux-tu analyser ? (ex: bitcoin, ethereum, dogecoin) 📉")
        return ASK_TOKEN
    else:
        await update.message.reply_text("🔴 Arrêt de l'analyse. À bientôt !")
        return ConversationHandler.END


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("👋 Quel(s) token(s) veux-tu analyser ? (ex: bitcoin, ethereum, dogecoin) 📉")
    return ASK_TOKEN

async def ask_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    tokens = [token.strip().lower() for token in update.message.text.split(',')]
    for token in tokens:
        await analyze_and_reply(update, token)
    return CONFIRM_CONTINUE

def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Ajouter un gestionnaire de callback pour gérer les réponses du bouton "Oui/Non"
    application.add_handler(CallbackQueryHandler(handle_continue))
    
    application.add_handler(CommandHandler("history", show_history))
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            ASK_TOKEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_token)],
            CONFIRM_CONTINUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_token)]
        },
        fallbacks=[]
    )
    
    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == '__main__':
    main()
