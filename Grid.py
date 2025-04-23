class TelegramInterface:
    def __init__(self, token, chat_id, bot_logic):
        self.token = token
        self.chat_id = chat_id
        self.bot = Bot(token=self.token)
        self.updater = Updater(token=self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.bot_logic = bot_logic

        self.dispatcher.add_handler(CommandHandler("status", self.status))
        self.dispatcher.add_handler(CommandHandler("pause", self.pause))
        self.dispatcher.add_handler(CommandHandler("resume", self.resume))
        self.dispatcher.add_handler(CommandHandler("capital", self.set_capital, pass_args=True))
        self.dispatcher.add_handler(CommandHandler("gridreport", self.grid_report))
        self.dispatcher.add_handler(CommandHandler("stop", self.stop))
        self.dispatcher.add_handler(CommandHandler("help", self.help))

    def send_message(self, message):
        self.bot.send_message(chat_id=self.chat_id, text=message)

    def status(self, update, context):
        etat = "✅ Actif" if self.bot_logic.running else "⏸ Pause"
        msg = (
            f"📊 Status Bot\n"
            f"Actif: {self.bot_logic.current_symbol or 'Aucun'}\n"
            f"Capital: {self.bot_logic.capital:.2f} USDT\n"
            f"Performance: +{(self.bot_logic.total_sell_revenue - self.bot_logic.total_buy_cost - self.bot_logic.total_fees):.2f} USDT\n"
            f"Etat: {etat}"
        )
        self.send_message(msg)

    def pause(self, update, context):
        self.bot_logic.running = False
        self.send_message("⏸ Bot mis en pause")

    def resume(self, update, context):
        self.bot_logic.running = True
        self.send_message("▶ Reprise des opérations")

    def stop(self, update, context):
        self.send_message("🛑 Arrêt complet du bot...")
        self.bot_logic.running = False
        os._exit(0)

    def help(self, update, context):
        help_msg = (
            "❓ Commandes disponibles:\n"
            "/status - Etat actuel\n"
            "/pause - Mettre en pause\n"
            "/resume - Reprendre\n"
            "/capital <montant> - Modifier capital\n"
            "/gridreport - Détails grille\n"
            "/stop - Arrêt complet\n"
            "/help - Aide"
        )
        self.send_message(help_msg)

    def set_capital(self, update, context):
        try:
            new_capital = float(context.args[0])
            self.bot_logic.capital = new_capital
            self.send_message(f"💰 Capital mis à jour: {new_capital} USDT")
        except:
            self.send_message("⚠ Usage: /capital <montant>")

    def grid_report(self, update, context):
        if self.bot_logic.grid_lower:
            msg = (
                f"📊 Rapport Grille\n"
                f"Symbole: {self.bot_logic.current_symbol}\n"
                f"Plage: {self.bot_logic.grid_lower:.2f} - {self.bot_logic.grid_upper:.2f}\n"
                f"Niveaux: {self.bot_logic.grid_levels}\n"
                f"Taille ordre: {self.bot_logic.order_amount:.6f}"
            )
        else:
            msg = "⚠ Grille non initialisée"
        self.send_message(msg)

    def start(self):  # Ligne 125 corrigée
        self.updater.start_polling()
