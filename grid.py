from hyperliquid.info import Info
from hyperliquid.utils import constants

# Remplace cette adresse par l'adresse Ethereum de l'utilisateur dont tu veux voir le solde
user_address = "0x7A58F1dDb718D28e9ea62c2fe13bf881e50B3421"

# Utilise l'URL de l'API principale (mainnet)
info = Info(constants.MAINNET_API_URL, skip_ws=True)

# Récupère l'état du clearinghouse spot pour l'utilisateur
user_state = info.user_state(user_address)

# Affiche les soldes
if "spotBalances" in user_state:
    print("Balances spot pour l'utilisateur", user_address)
    for balance in user_state["spotBalances"]:
        coin = balance.get("coin")
        total = balance.get("total")
        print(f"{coin}: {total}")
else:
    print("Aucun solde spot trouvé pour cet utilisateur.")
