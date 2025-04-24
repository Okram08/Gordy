require("dotenv").config();
const { ethers } = require("ethers");
const axios = require("axios");

// 🔐 Clés privées
const privateKey = process.env.HL_PRIVATE_KEY;
const walletAddress = process.env.HL_WALLET_ADDRESS;

if (!privateKey || !walletAddress) {
  console.error("❌ HL_PRIVATE_KEY ou HL_WALLET_ADDRESS manquant dans .env");
  process.exit(1);
}

// 🧠 Préparation du message à signer
const message = `Register vault address ${walletAddress}`;
const signMessage = async () => {
  try {
    const wallet = new ethers.Wallet(privateKey);
    const signature = await wallet.signMessage(message);

    const payload = {
      address: walletAddress,
      msg: message,
      sig: signature,
    };

    console.log(`📦 Enregistrement de : ${walletAddress}`);
    const response = await axios.post("https://api.hyperliquid.xyz/register", payload);

    if (response.data) {
      console.log("✅ Vault enregistré avec succès !");
      console.log(response.data);
    } else {
      console.error("❌ Erreur : Réponse vide ou invalide");
    }
  } catch (error) {
    console.error("❌ Erreur lors de l'enregistrement :", error.message);
  }
};

signMessage();
