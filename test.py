import os
import requests
import json
from dotenv import load_dotenv
from eth_account import Account

# 🔐 Chargement des variables d'environnement
load_dotenv()
PRIVATE_KEY = os.getenv("HL_PRIVATE_KEY")
WALLET_ADDRESS = os.getenv("HL_WALLET_ADDRESS")

# 📤 Signature de la requête
def sign_register_payload(private_key, wallet_address):
    msg = f"Register vault address {wallet_address}"
    signed_message = Account.sign_message(
        encode_defunct(text=msg), private_key=private_key
    )
    return {
        "address": wallet_address,
        "msg": msg,
        "sig": signed_message.signature.hex()
    }

# 📦 Enregistrement
def register_vault():
    url = "https://api.hyperliquid.xyz/register"
    payload = sign_register_payload(PRIVATE_KEY, WALLET_ADDRESS)

    headers = {
        "Content-Type": "application/json"
    }

    try:
        print(f"📦 Enregistrement de : {WALLET_ADDRESS}")
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            print("✅ Vault enregistré avec succès !")
            print(response.json())
        else:
            print(f"❌ Erreur {response.status_code} : {response.text}")
    except Exception as e:
        print(f"❌ Exception : {e}")

# 👟 Lancement
if __name__ == "__main__":
    from eth_account.messages import encode_defunct
    register_vault()
