require('dotenv').config();
const { Hyperliquid } = require('hyperliquid');
const prompt = require('prompt-sync')({ sigint: true });

const privateKey = process.env.HL_PRIVATE_KEY;
if (!privateKey) {
  console.error('Erreur : Clé privée manquante dans .env');
  process.exit(1);
}

const sdk = new Hyperliquid({
  privateKey: privateKey,
  enableWs: false,
  testnet: false,
});

async function getCurrentPrice(symbol) {
  const allMids = await sdk.info.getAllMids();
  const mids = allMids.spotMids;
  const spotSymbol = symbol.replace('-SPOT', '/USDC');
  if (!mids[spotSymbol]) {
    throw new Error(`Prix spot non trouvé pour le symbole ${spotSymbol}`);
  }
  return parseFloat(mids[spotSymbol]);
}

async function placeGridOrders(symbol, lower, upper, grids, totalAmount) {
  const step = (upper - lower) / (grids - 1);
  const orders = [];
  const amountPerOrder = totalAmount / grids;

  for (let i = 0; i < grids; i++) {
    const price = lower + step * i;
    const side = price < (lower + upper) / 2 ? 'buy' : 'sell';
    orders.push({
      symbol,
      price: price.toFixed(2),
      size: amountPerOrder.toFixed(6),
      side,
      type: 'limit',
    });
  }

  for (const order of orders) {
    try {
      const res = await sdk.order.placeOrder(order);
      console.log(`Ordre ${order.side} placé à ${order.price} :`, res);
    } catch (err) {
      console.error('Erreur lors de la pose de l\'ordre :', err);
    }
  }
}

async function main() {
  const symbol = prompt('Entrez le symbole spot (ex: BTC-SPOT) : ').trim();
  const totalAmount = parseFloat(prompt('Montant total à allouer (en USDC) : ').trim());
  const grids = parseInt(prompt('Nombre de grilles (ex: 7) : ').trim(), 10);

  const currentPrice = await getCurrentPrice(symbol);

  // Fourchette automatique : ±3% autour du prix actuel
  const rangePct = 0.03;
  const lower = currentPrice * (1 - rangePct);
  const upper = currentPrice * (1 + rangePct);

  console.log(`Prix actuel : ${currentPrice}`);
  console.log(`Fourchette automatique : [${lower.toFixed(2)} ; ${upper.toFixed(2)}]`);

  await placeGridOrders(symbol, lower, upper, grids, totalAmount);
}

main();
