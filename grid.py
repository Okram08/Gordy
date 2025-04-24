import requests
import json

def get_spot_balance(address):
    # URL de l'API
    url = "https://api.hyperliquid.xyz/info"
    
    # Corps de la requête
    payload = {
        "type": "spotClearinghouseState",
        "user": address
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Envoi de la requête POST
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        # Vérification du statut de la réponse
        if response.status_code == 200:
            # Traitement de la réponse
            data = response.json()
            
            # Affichage des balances
            if "balances" in data:
                print(f"Balances spot pour l'utilisateur {address}:")
                for balance in data["balances"]:
                    coin = balance.get("coin")
                    total = balance.get("total")
                    print(f"{coin}: {total}")
            else:
                print("Aucun solde spot trouvé pour cet utilisateur.")
        else:
            print(f"Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Erreur lors de la requête : {e}")

# Remplace cette adresse par celle dont tu veux voir le solde
user_address = "0x7A58F1dDb718D28e9ea62c2fe13bf881e50B3421"

get_spot_balance(user_address)
