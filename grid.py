from hyperliquid.info import Info
from hyperliquid.utils import constants

# Remplace ici par une vraie adresse Ethereum ayant un solde sur Hyperliquid
user_address = "0x7A58F1dDb718D28e9ea62c2fe13bf881e50B3421"

info = Info(constants.MAINNET_API_URL, skip_ws=True)
user_state = info.user_state(user_address)

# Affiche tous les champs pour debug
print("Réponse brute de l'API :")
print(user_state)

# Cherche le champ de balances
spot_balances = user_state.get("spotBalances") or user_state.get("balances")

if spot_balances:
    print(f"\nBalances spot pour l'utilisateur {user_address}:")
    for balance in spot_balances:
        coin = balance.get("coin")
        total = balance.get("total")
        print(f"{coin}: {total}")
else:
    print("Aucun solde spot trouvé pour cet utilisateur.")
