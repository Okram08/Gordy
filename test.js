require("dotenv").config();
const { Hyperliquid } = require("hyperliquid");

const HL_WALLET_ADDRESS = process.env.HL_WALLET_ADDRESS;
const HL_PRIVATE_KEY = process.env.HL_PRIVATE_KEY;

async function registerVault() {
  try {
    const exchange = new Hyperliquid({
      wallet: {
        address: HL_WALLET_ADDRESS,
        privateKey: HL_PRIVATE_KEY,
      },
      baseUrl: "https://api.hyperliquid.xyz",
    });

    console.log("📦 Enregistrement du vault pour :", HL_WALLET_ADDRESS);

    const result = await exchange.register();
    console.log("✅ Résultat :", result);
  } catch (err) {
    console.error("❌ Erreur lors de l'enregistrement :", err.message || err);
  }
}

registerVault();
