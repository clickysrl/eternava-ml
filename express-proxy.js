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

// --- PING CHECK (Circuit Breaker) ---
function pingPort(host, port, timeoutMs = 800) {
    return new Promise((resolve, reject) => {
        const s = net.connect({ host, port });
        const t = setTimeout(() => { s.destroy(); reject(new Error('timeout')); }, timeoutMs);
        s.once('connect', () => { clearTimeout(t); s.destroy(); resolve(true); });
        s.once('error', (e) => { clearTimeout(t); reject(e); });
    });
}

// Stato globale del circuit breaker
let downUntil = 0;

// Middleware per HTTP
async function fastFailGuard(req, res, next) {
    const now = Date.now();
    if (now < downUntil) {
        return res.status(502).json({ ok: false, error: 'ml_unreachable_cache' });
    }
    try {
        await pingPort(TARGET_HOST, TARGET_PORT, 800);
        next();
    } catch (e) {
        downUntil = now + 2500; // Circuit breaker: stop per 2.5s
        res.status(502).json({ ok: false, error: 'ml_unreachable', detail: e.message });
    }
}

// --- PROXY CONFIG ---
const mlProxy = createProxyMiddleware({
    target: `http://${TARGET_HOST}:${TARGET_PORT}`,
    changeOrigin: true,
    ws: true, // Abilita WS
    xfwd: true,
    proxyTimeout: 10 * 60 * 1000,
    timeout: 10 * 60 * 1000,

    // GESTORE ERRORI IBRIDO (HTTP + WS)
    onError(err, req, res) {
        const isWebSocket = !res.writeHead; // Se non ha writeHead, è un socket (WS)

        if (isWebSocket) {
            // Errore su WebSocket: Chiudiamo il socket pulito
            console.error('[Proxy WS Error]', err.message);
            try {
                res.write('HTTP/1.1 502 Bad Gateway\r\n\r\n');
                res.end();
            } catch (e) { }
        } else {
            // Errore su HTTP Standard
            if (!res.headersSent) {
                res.writeHead(502, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ ok: false, error: 'ml_proxy_error', detail: err.message }));
            }
        }
    }
});

// Usa il proxy per le richieste HTTP
app.use('/', fastFailGuard, mlProxy);

// --- HTTPS SERVER ---
try {
    const tlsOptions = {
        key: fs.readFileSync(KEY_PATH),
        cert: fs.readFileSync(CERT_LEAF),
        ca: [fs.readFileSync(CA_1), fs.readFileSync(CA_2)],
    };

    const server = https.createServer(tlsOptions, app);
    server.setTimeout(10 * 60 * 1000);

    // --- GESTIONE WEBSOCKET MANUALE (CON SICUREZZA) ---
    server.on('upgrade', async (req, socket, head) => {
        // 1. Controllo Circuit Breaker anche per WebSocket!
        const now = Date.now();
        if (now < downUntil) {
            socket.write('HTTP/1.1 502 Bad Gateway\r\n\r\n');
            socket.destroy();
            return;
        }

        // 2. Ping Rapido prima di passare la connessione
        try {
            await pingPort(TARGET_HOST, TARGET_PORT, 800);
            // 3. Se il server è vivo, delega al proxy
            mlProxy.upgrade(req, socket, head);
        } catch (e) {
            downUntil = now + 2500; // Attiva circuit breaker
            socket.write('HTTP/1.1 502 Bad Gateway\r\n\r\n');
            socket.destroy();
        }
    });

    server.listen(443, () => {
        console.log(`Express TLS proxy on :443 (Secure WSS) → http://${TARGET_HOST}:${TARGET_PORT}`);
    });

} catch (err) {
    console.error("Errore avvio server HTTPS:", err.message);
}