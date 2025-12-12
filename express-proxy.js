// /root/ml-serve/express-proxy.js
const fs = require('fs');
const https = require('https');
const net = require('net');
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');

// === Certificati ===
const CERT_LEAF = '/root/certs/nexipse.it_ssl_certificate.cer';
const KEY_PATH = '/root/certs/_.nexipse.it_private_key.key';
const CA_1 = '/root/certs/_.nexipse.it_ssl_certificate_INTERMEDIATE/intermediate1.cer';
const CA_2 = '/root/certs/_.nexipse.it_ssl_certificate_INTERMEDIATE/intermediate2.cer';

// === Target FastAPI locale ===
const TARGET_HOST = '127.0.0.1';
const TARGET_PORT = 9000;

const app = express();
app.disable('x-powered-by');

// --- FAST FAIL: ping TCP al target prima di proxyare ---
// se la porta è giù → 502 immediato (niente pending)
function pingPort(host, port, timeoutMs = 800) {
    return new Promise((resolve, reject) => {
        const s = net.connect({ host, port });
        const t = setTimeout(() => { s.destroy(); reject(new Error('timeout')); }, timeoutMs);
        s.once('connect', () => { clearTimeout(t); s.destroy(); resolve(true); });
        s.once('error', (e) => { clearTimeout(t); reject(e); });
    });
}

// mini circuit-breaker per non tempestare quando è giù
let downUntil = 0;
async function fastFailGuard(req, res, next) {
    const now = Date.now();
    if (now < downUntil) {
        res.status(502).json({ ok: false, error: 'ml_unreachable_cache' });
        return;
    }
    try {
        await pingPort(TARGET_HOST, TARGET_PORT, 800); // < 1s
        next();
    } catch (e) {
        downUntil = now + 2500; // 2.5s
        res
            .status(502)
            .json({ ok: false, error: 'ml_unreachable', detail: e.code || e.message });
    }
}

// --- Proxy vero e proprio ---
app.use(
    '/',
    fastFailGuard,
    createProxyMiddleware({
        target: `http://${TARGET_HOST}:${TARGET_PORT}`,
        changeOrigin: true,
        ws: false,
        xfwd: true,

        // timeout per la CONNESSIONE/INATTIVITÀ verso il target già connesso
        // (per upload lunghi lasciamo amplio; il fast-fail pensa al caso "server giù")
        proxyTimeout: 10 * 60 * 1000,
        timeout: 10 * 60 * 1000,

        // rispondi SUBITO con 502 in caso di ECONNREFUSED/ETIMEDOUT ecc.
        onError(err, req, res) {
            if (res.headersSent) return;
            res.writeHead(502, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
            res.end(JSON.stringify({
                ok: false,
                error: 'ml_unreachable',
                detail: err.code || String(err)
            }));
        },
    })
);

// --- HTTPS server ---
const tlsOptions = {
    key: fs.readFileSync(KEY_PATH),
    cert: fs.readFileSync(CERT_LEAF),
    ca: [fs.readFileSync(CA_1), fs.readFileSync(CA_2)],
};

const server = https.createServer(tlsOptions, app);
// tieni lunghi per upload; il fail-fast gestisce il caso "spento"
server.setTimeout(10 * 60 * 1000);

server.listen(443, () => {
    console.log(`Express TLS proxy on :443 → http://${TARGET_HOST}:${TARGET_PORT}`);
});
