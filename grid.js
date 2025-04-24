require('dotenv').config();
const { Hyperliquid } = require('hyperliquid');
const readline = require('readline');

// Paramètres de la grille
const GRID_LEVELS = 3;
const GRID_SPREAD = 0.01; // 1% au-dessus et en dessous du prix spot

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
    // symbol: "BTC-USDC" => market: "BTC"
    const market = symbol.split('-')[0];
    const info = await sdk.info.markets();
    const mkt = info.find(m => m.name === market);
    if (!mkt) throw new Error("Marché non trouvé");
    return parseFloat(mkt.markPrice);
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
    // symbol: "BTC-USDC" => market: "BTC"
    const market = symbol.split('-')[0];
    const order = {
        market,
        side, // "buy" ou "sell"
        size: quantity,
        price,
        type: "limit"
    };
    console.log(`🚀 Envoi de l'ordre LIVE:`, order);
    const res = await sdk.trade.placeOrder(order);
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
        const quantity = allocations[i] / price;
        await placeOrder(symbol, "buy", price, quantity);
        await new Promise(r => setTimeout(r, 1000));
    }
}

main();
