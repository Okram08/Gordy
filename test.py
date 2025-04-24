import os
from dotenv import load_dotenv
from web3 import Web3
import requests

load_dotenv()  # Charge les variables du .env

class GridBot:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(os.getenv('ALCHEMY_RPC_URL')))
        self.private_key = os.getenv('PRIVATE_KEY')
        self.router_address = Web3.to_checksum_address(os.getenv('ROUTER_ADDRESS'))
        
        # Initialisation du contrat 1inch
        self.router_abi = self._get_1inch_abi()
        self.contract = self.w3.eth.contract(
            address=self.router_address,
            abi=self.router_abi
        )

    def _get_1inch_abi(self):
        """Récupère l'ABI actuel du routeur 1inch"""
        url = f"https://api.1inch.io/v5.0/{os.getenv('CHAIN_ID')}/protocols"
        return requests.get(url).json()['protocols']['1inch']['abi']

    def create_swap_transaction(self, from_token, to_token, amount):
        """Crée une transaction de swap via l'API 1inch"""
        url = f"https://api.1inch.io/v5.0/{os.getenv('CHAIN_ID')}/swap" \
              f"?fromTokenAddress={from_token}" \
              f"&toTokenAddress={to_token}" \
              f"&amount={amount}" \
              f"&fromAddress={os.getenv('WALLET_ADDRESS')}" \
              "&slippage=1"
        
        response = requests.get(url).json()
        return response['tx']

    def execute_grid_order(self, order_type):
        """Exécute un ordre de la grille"""
        base_token = os.getenv('BASE_TOKEN')
        quote_token = os.getenv('QUOTE_TOKEN')
        
        tx_data = self.create_swap_transaction(
            from_token=base_token if order_type == 'SELL' else quote_token,
            to_token=quote_token if order_type == 'SELL' else base_token,
            amount=int(float(os.getenv('ORDER_AMOUNT')) * 1e6)  # Adaptez la décimale
        )

        transaction = {
            'chainId': int(os.getenv('CHAIN_ID')),
            'to': self.router_address,
            'data': tx_data['data'],
            'value': int(tx_data.get('value', 0)),
            'gas': int(tx_data['gas']),
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(os.getenv('WALLET_ADDRESS'))
        }

        signed_tx = self.w3.eth.account.sign_transaction(transaction, self.private_key)
        return self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

# Exemple d'utilisation
if __name__ == "__main__":
    bot = GridBot()
    bot.execute_grid_order('BUY')
