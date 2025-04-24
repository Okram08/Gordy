require('dotenv').config();
const { Hyperliquid } = require('hyperliquid');

const sdk = new Hyperliquid({
    privateKey: process.env.HL_PRIVATE_KEY,
    walletAddress: process.env.HL_WALLET_ADDRESS
});

(async () => {
    try {
        const res = await sdk.exchange.placeOrder({
            coin: 'UBTC-SPOT',
            is_buy: true,
            sz: '0.0001',
            limit_px: 10000,
            reduce_only: false,
            order_type: { limit: { tif: 'Gtc' } }
        });
        console.log(res);
    } catch (e) {
        console.error(e);
    }
})();
