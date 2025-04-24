require('dotenv').config();

const { Hyperliquid } = require('hyperliquid');
const prompt = require('prompt-sync')({ sigint: true });

// Récupération de la clé privée depuis .env
const privateKey = process.env.HL_PRIVATE_KEY;
if (!privateKey) {
  console.error('Erreur : La clé privée Hyperliquid n\'est pas définie dans .env (HL_PRIVATE_KEY)');
  process.exit(1);
}

// Demande des paramètres à l'utilisateur
const symbol = prompt('Entrez le symbole spot (ex: BTC-SPOT) : ').trim();
const montant = prompt('Montant à allouer (en USDC) : ').trim();
const price = prompt('Prix limite souhaité : ').trim();

// Initialisation du SDK Hyperliquid
const sdk = new Hyperliquid({
  privateKey: privateKey,
  enableWs: false,
  testnet: false, // Passe à true si tu veux utiliser le testnet
});

async function main() {
  try {
    // Affichage des soldes (optionnel)
    const balances = await sdk.info.userState();
    console.log('Vos soldes :', balances);

    // Passage de l'ordre spot
    const order = await sdk.order.placeOrder({
      symbol: symbol,  // ex: 'BTC-SPOT'
      price: price,    // prix limite
      size: montant,   // montant en USDC
      side: 'buy',     // 'buy' ou 'sell'
      type: 'limit',   // 'limit' ou 'market'
    });

    console.log('Ordre envoyé :', order);
  } catch (err) {
    console.error('Erreur lors de la passation de l\'ordre :', err);
  }
}

main();
