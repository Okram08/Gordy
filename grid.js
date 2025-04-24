require('dotenv').config();
const { Hyperliquid } = require('hyperliquid');
const readline = require('readline');

// Paramètres de la grille
const GRID_LEVELS = 3;
const GRID_SPREAD = 0.01; // 1% au-dessus et en dessous du prix spot

// Mapping pour les tokens connus
const COIN_MAP = {
    "BTC": "UBTC",
    "BTC-USDC": "UBTC-USDC",
    // Ajoute d'autres mappings si besoin
};

function getCoinFromSymbol(symbol) {
    const base = symbol.split('-')[0].toUpperCase();
    return COIN_MAP[base] || base;
}

function roundTo8(x) {
    // Arrondit à 8 décimales, supprime les zéros inutiles
    return Number.parseFloat(x).toFixed(8).replace(/\.?0+$/, '');
}

const sdk = new Hyperliquid({
    privateKey: process.env.HL_PRIVATE_KEY,
    walletAddress: process.env.HL_WALLET_ADDRESS
});

function ask(question) {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    return new Promise(resolve => rl.question(question, ans => {
        rl.close();
        resolve(ans);
    }));
}

async function getSpotPrice(symbol) {
    const coin = getCoinFromSymbol(symbol);
    const allMids = await sdk.info.getAllMids();
    const keys = Object.keys(allMids);

    // Affiche tous les marchés disponibles pour debug
    console.log("Marchés disponibles :", keys.join(', '));

    // Recherche d'une clé qui commence par le coin et contient 'SPOT'
    const spotKey = keys.find(
        k => k.toUpperCase().startsWith(coin) && k.toUpperCase().includes('SPOT')
    );
    if (!spotKey) {
        throw new Error("Marché spot non trouvé pour " + coin + ". Clés disponibles : " + keys.join(', '));
    }
    return parseFloat(allMids[spotKey]);
}

function buildGrid(centerPrice, levels, spread) {
    const prices = [];
    for (let i = 0; i < levels; i++) {
        const offset = (i - Math.floor(levels / 2)) * spread;
        const gridPrice = centerPrice * (1 + offset);
        prices.push(Number(gridPrice.toFixed(2)));
    }
    return prices;
}

function distributeCapital(capital, levels) {
    const amountPerLevel = capital / levels;
    return Array(levels).fill(Number(amountPerLevel.toFixed(6)));
}

async function placeOrder(symbol, side, price, quantity) {
    const coin = getCoinFromSymbol(symbol);
    const allMids = await sdk.info.getAllMids();
    const keys = Object.keys(allMids);
    const spotKey = keys.find(
        k => k.toUpperCase().startsWith(coin) && k.toUpperCase().includes('SPOT')
    );
    if (!spotKey) {
        throw new Error("Impossible de trouver la clé spot pour la paire " + symbol);
    }
    const order = {
        coin: spotKey,
        is_buy: side === "buy",
        sz: quantity,
        limit_px: price,
        reduce_only: false,
        order_type: { limit: { tif: 'Gtc' } }
    };
    console.log(`🚀 Envoi de l'ordre LIVE:`, order);
    const res = await sdk.exchange.placeOrder(order);
    if (res.status === "ok") {
        console.log("✅ Ordre envoyé avec succès:", res);
    } else {
        console.log("❌ Erreur lors de l'envoi de l'ordre:", res);
    }
    return res;
}

async function main() {
    console.log("🔁 Lancement du Grid Trading Bot (Hyperliquid SDK Node.js)...");

    const symbol = (await ask("🪙 Quelle paire veux-tu trader ? (ex: BTC-USDC) : ")).toUpperCase();
    const totalCapital = parseFloat(await ask("💰 Capital à allouer (en USDC) : "));

    let spotPrice;
    try {
        spotPrice = await getSpotPrice(symbol);
    } catch (e) {
        console.error("❌ Impossible de récupérer le prix spot :", e.message);
        process.exit(1);
    }
    console.log(`📈 Prix spot actuel pour ${symbol} : ${spotPrice} USDC`);

    const grid = buildGrid(spotPrice, GRID_LEVELS, GRID_SPREAD);
    const allocations = distributeCapital(totalCapital, GRID_LEVELS);

    console.log("\n📋 Stratégie Grid Trading :");
    for (let i = 0; i < GRID_LEVELS; i++) {
        console.log(`Grille ${i+1}: Prix ${grid[i]} USDC → Allocation ${allocations[i]} USDC`);
    }

    const confirmation = (await ask("\n✅ Confirmer le placement des ordres (LIVE) ? (o/n) : ")).toLowerCase();
    if (confirmation !== "o") {
        console.log("❌ Annulé par l'utilisateur.");
        process.exit(0);
    }

    for (let i = 0; i < GRID_LEVELS; i++) {
        const price = grid[i];
        let quantity = allocations[i] / price;
        quantity = roundTo8(quantity);
        await placeOrder(symbol, "buy", price, quantity);
        await new Promise(r => setTimeout(r, 1000));
    }
}

main();
