# /root/ml-serve/app.py
import os, asyncio, tempfile, uuid, hmac, hashlib, subprocess, json, shutil
from typing import Optional, List, Tuple, Dict
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
from TTS.api import TTS
import requests
import librosa
import re
import math
from voicecheck import (
    load_audio_mono_16k,
    vad_segments,
    cut_chunks,
    embed_chunks,
    speaker_centroid_from_paths,
    grade_clip,
    analyze_file,
    build_profile_bank
)
import yt_dlp
import shutil
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# --- CONFIGURAZIONE ---
# URL del server Node (cambia con l'indirizzo reale in produzione)
NODE_API_URL = os.getenv("NODE_API_URL", "https://api.eternava.it/api")

# Cartelle
XTTS_STORE_DIR = os.getenv("XTTS_SPEAKER_DIR", "/data/xtts_speakers")
PUBLIC_AUDIO_DIR = "/data/public_audio" # Cartella pubblica per i file audio

os.makedirs(XTTS_STORE_DIR, exist_ok=True)
os.makedirs(PUBLIC_AUDIO_DIR, exist_ok=True)
_ABBR_IT = {
    "sig", "sigg", "sigra", "sig.na", "sig.ra", "dott", "dr", "ing", "prof", "avv", "ecc",
    "cap", "gen", "comm", "rag", "arch"
}


# =========================
# ========= APP ===========
# =========================

app = FastAPI(title="ML Transcription/TTS Service")

def _centroid_from_file(path: str) -> np.ndarray:
    wav, sr = load_audio_mono_16k(path)
    segs = vad_segments(wav, sr, frame_ms=30, aggressiveness=2, min_speech_ms=300, max_gap_ms=300)
    chunks = cut_chunks(wav, sr, segs, max_chunk_sec=5.0)
    embs = embed_chunks(chunks, sr=sr)
    if embs.shape[0] == 0:
        return np.zeros((192,), dtype=np.float32)
    c = np.mean(embs, axis=0)
    c /= (np.linalg.norm(c) + 1e-9)
    return c.astype(np.float32)
def _debug_check_refs(refs: List[str]):
    import torchaudio
    bad = []
    for p in refs:
        try:
            torchaudio.load(p)
        except Exception as e:
            print(f"[XTTS] BAD REF: {p} -> {e}")
            bad.append(p)
    return bad

@app.post("/v1/voice/compare")
async def voice_compare(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    _require_auth(authorization)
    pA = await _save_temp(file_a)
    pB = await _save_temp(file_b)
    try:
        cA = _centroid_from_file(pA)
        cB = _centroid_from_file(pB)
        cos = float(np.dot(cA, cB))
        return {"ok": True, "cosine": round(cos, 3), "similarity_pct": round(cos * 100.0, 1)}
    finally:
        try: os.remove(pA)
        except: pass
        try: os.remove(pB)
        except: pass

def normalize_text_it(t: str) -> str:
    s = (t or "").strip()

    # ellissi -> pausa media (non troppo lunga)
    s = re.sub(r"\.{3,}", " … ", s)  # lascio l'ellissi come segnale di pausa, ma la capperemo in audio

    # numeri decimali: 3.14 -> "3 virgola 14"
    s = re.sub(r"(?<=\d)\.(?=\d)", " virgola ", s)

    # >>> PATCH FIX: rimuovi SOLO il punto dopo abbreviazioni note
    # prima: cancellavi l'intera abbreviazione!
    def _strip_dot_after_abbr(m):
        word = m.group(1)
        # normalizza 'Sig.' -> 'sig'
        key = word.lower()
        return word if key in _ABBR_IT else word + "."

    s = re.sub(r"\b([A-Za-z]{1,6})\.", _strip_dot_after_abbr, s)

    # "punto"/"virgola" DETTATO (capita da trascrizioni) -> segni veri
    # sostituisco quando isolati tra parole, non tocco se fanno parte del contenuto
    s = re.sub(r"\b[Pp]unto\b", ".", s)
    s = re.sub(r"\b[Vv]irgola\b", ",", s)
    s = re.sub(r"\b[Dd]ue\s+punti\b", ":", s)
    s = re.sub(r"\b[Pp]unto\s+e\s+a\s+capo\b", "", s)  # safety per casi strani

    # punti normali -> virgola (pausa breve)  | evita i casi già gestiti sopra
    s = re.sub(r"(?<!\d)\.(?!\d)", ", ", s)

    # doppia punteggiatura -> compatta
    s = re.sub(r"\s*([,;:!?])\s*", r"\1 ", s)

    # compatta spazi
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s
# =========================
# ======== CONFIG =========
# =========================

# ---- XTTS (voice clone) ----
# ---- XTTS (voice clone) ----
XTTS_MODEL_NAME = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
XTTS_DEVICE = os.getenv("XTTS_DEVICE", "cuda")
XTTS_SR = int((os.getenv("XTTS_SAMPLE_RATE", "24000") or "24000").strip())
XTTS_STORE_DIR = os.getenv("XTTS_SPEAKER_DIR", "/data/xtts_speakers")

ENROLL_MAX_FILES = int((os.getenv("XTTS_ENROLL_MAX_FILES", "500") or "500").strip())
ENROLL_MAX_PER_FILE_SEC = int((os.getenv("XTTS_ENROLL_MAX_PER_FILE_SEC","10") or "10").strip())
ENROLL_MAX_FILE_DURATION_SEC = int((os.getenv("XTTS_ENROLL_MAX_FILE_DURATION_SEC","60000") or "60000").strip())

MAX_TTS_CONCURRENT = int((os.getenv("MAX_TTS_CONCURRENT","1") or "1").strip())

# >>> PATCH: nuovi toggle
XTTS_SPLIT_SENTENCES = os.getenv("XTTS_SPLIT_SENTENCES", "false").lower() == "true"
XTTS_MAX_SIL_MS = int((os.getenv("XTTS_MAX_SIL_MS", "400") or "400").strip())  # cap silenzi nell'audio
XTTS_NOISE_TRIM_ENROLL = os.getenv("XTTS_NOISE_TRIM_ENROLL", "true").lower() == "true"

XTTS_CHUNK_CHARS = int((os.getenv("XTTS_CHUNK_CHARS", "260") or "260").strip())       # max caratteri per pezzo
XTTS_CHUNK_XFADE_MS = int((os.getenv("XTTS_CHUNK_XFADE_MS", "40") or "40").strip())   # cross-fade tra pezzi (ms)
# ---- ASR (transcribe) ----
MODEL_SIZE       = os.getenv("MODEL_SIZE", "large-v3")
DEVICE           = os.getenv("DEVICE", "cuda")                 # GPU per Whisper
DEFAULT_CTYPE    = os.getenv("COMPUTE_TYPE", "float16")        # qualità × velocità
FALLBACK_CTYPE   = os.getenv("FALLBACK_COMPUTE_TYPE", "int8_float16")
MAX_CONCURRENT   = int((os.getenv("MAX_CONCURRENT", "1") or "1").strip())  # niente commenti in linea nello .env
DEFAULT_LANGUAGE = os.getenv("LANGUAGE")
VAD_FILTER       = os.getenv("VAD_FILTER", "true").lower() == "true"

DEFAULT_QUALITY  = os.getenv("PROFILE", "balanced").lower()    # "fast" | "balanced" | "accurate"
MAX_FILE_MB      = int((os.getenv("MAX_FILE_MB", "300") or "300").strip())  # 0 = nessun limite
INITIAL_PROMPT   = os.getenv("INITIAL_PROMPT") or None
FORCE_CHUNK      = int((os.getenv("CHUNK_LENGTH", "0") or "0").strip())     # 0 = auto

def _ffmpeg_split_mono_24k(src: str, out_dir: str, seg_sec: int) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    base = f"ref_{uuid.uuid4().hex}_%03d.wav"
    pattern = os.path.join(out_dir, base)

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-y", "-i", src,
        "-vn", "-sn", "-dn",          # ignora video/sottotitoli/data
        "-map", "a:0",                # prima traccia audio
        "-ar", "24000", "-ac", "1",
        "-c:a", "pcm_s16le",
    ]
    if XTTS_NOISE_TRIM_ENROLL:
        cmd += ["-af", "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-45dB:"
                        "stop_periods=1:stop_duration=0.2:stop_threshold=-45dB"]

    cmd += [
        "-f", "segment", "-segment_time", str(seg_sec),
        "-reset_timestamps", "1",
        # opzionale per non saturare CPU:
        # "-threads", str(os.cpu_count() // 2 or 1),
        pattern
    ]

    subprocess.run(cmd, check=True)  # <-- niente capture_output
    files = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if re.fullmatch(r"ref_[0-9a-f]+_\d{3}\.wav", f)
    )
    return files


# ---- Auth ----
def _split_env_list(name): return [x.strip() for x in (os.getenv(name, "")).split(",") if x.strip()]
TOKENS_PLAIN  = set(_split_env_list("ML_API_TOKENS"))
if os.getenv("ML_API_TOKEN"): TOKENS_PLAIN.add(os.getenv("ML_API_TOKEN"))
TOKENS_SHA256 = set(_split_env_list("ML_API_TOKENS_SHA256"))
if os.getenv("ML_API_TOKEN_SHA256"): TOKENS_SHA256.add(os.getenv("ML_API_TOKEN_SHA256"))

def _sha256(s: str) -> str: return hashlib.sha256(s.encode("utf-8")).hexdigest()
def _auth_ok(bearer: str) -> bool:
    if not bearer: return False
    for t in TOKENS_PLAIN:
        if t and hmac.compare_digest(bearer, t): return True
    b_hash = _sha256(bearer)
    for h in TOKENS_SHA256:
        if h and hmac.compare_digest(b_hash, h): return True
    return False

def _require_auth(authorization: Optional[str]):
    if not TOKENS_PLAIN and not TOKENS_SHA256:
        raise HTTPException(status_code=500, detail="Server misconfigured: no API token set")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if not _auth_ok(authorization.split(" ",1)[1]):
        raise HTTPException(status_code=403, detail="Invalid token")

# ---- Semafori concorrenza ----
sem = asyncio.Semaphore(MAX_CONCURRENT)            # ASR
tts_sem = asyncio.Semaphore(MAX_TTS_CONCURRENT)    # TTS
MAX_SUM_CONCURRENT = int((os.getenv("MAX_SUM_CONCURRENT", "2") or "2").strip())
sum_sem = asyncio.Semaphore(MAX_SUM_CONCURRENT)
USE_LLM = os.getenv("SUM_BACKEND", "none").lower() == "llm"
if USE_LLM:
    from summarizer_llm import summarize as _do_summarize
else:
    # fallback no-LLM (opzionale): echo
    def _do_summarize(text: str, lang: str = "it", style: str = "paragraph", max_words: int = 180):
        from dataclasses import dataclass
        @dataclass
        class S: summary: str; bullets: list|None; model: str
        return S(summary=text[:max_words*6], bullets=None, model="noop")

# =========================
# ====== XTTS UTILS =======
# =========================
# ---- ChatGPT (OpenAI) per modalità chat (risposta testuale con sessione per utente) ----
CHAT_OPENAI_BASE = os.getenv("CHAT_OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
CHAT_OPENAI_KEY = os.getenv("CHAT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
CHAT_OPENAI_MODEL = os.getenv("CHAT_OPENAI_MODEL", "gpt-5-mini")
CHAT_TEMP = float(os.getenv("CHAT_TEMP", "0.5"))
CHAT_TOP_P = float(os.getenv("CHAT_TOP_P", "0.9"))
CHAT_MAX_TOKENS = int((os.getenv("CHAT_MAX_TOKENS", "512") or "512").strip())
CHAT_MAX_MESSAGES = int((os.getenv("CHAT_MAX_MESSAGES", "10") or "10").strip())
CHAT_MAX_HISTORY_CHARS = int((os.getenv("CHAT_MAX_HISTORY_CHARS", "6000") or "6000").strip())
CHAT_LANG_DEFAULT = os.getenv("CHAT_LANG_DEFAULT", "it")

# mappa in-memory: user_id -> sessione {lang, messages}
_chat_sessions: Dict[str, Dict] = {}


def _chat_system_message(lang: str) -> Dict[str, str]:
    L = (lang or CHAT_LANG_DEFAULT or "it").lower()
    if L.startswith("it"):
        label = "italiano"
    else:
        label = L
    return {
        "role": "system",
        "content": f"Sei un assistente vocale. Rispondi in modo colloquiale, naturale e abbastanza breve in lingua {label}.",
    }


def _get_chat_session(user_id: Optional[str], lang: str) -> Tuple[str, Dict]:
    uid = (user_id or "anon").strip() or "anon"
    uid = uid[:64]  # limite per field 'user' di OpenAI

    sess = _chat_sessions.get(uid)
    if sess is None:
        sess = {
            "lang": lang or CHAT_LANG_DEFAULT or "it",
            "messages": [_chat_system_message(lang)],
        }
    else:
        # se cambia lingua, aggiorno il system message
        if lang and sess.get("lang") != lang:
            sess["lang"] = lang
            if sess.get("messages") and sess["messages"][0].get("role") == "system":
                sess["messages"][0] = _chat_system_message(lang)
            else:
                sess["messages"].insert(0, _chat_system_message(lang))

    _chat_sessions[uid] = sess
    return uid, sess


def _trim_chat_session(sess: Dict):
    msgs = sess.get("messages") or []
    if not msgs:
        msgs = [_chat_system_message(sess.get("lang") or CHAT_LANG_DEFAULT or "it")]

    # il primo messaggio deve essere system
    first = msgs[0]
    if first.get("role") != "system":
        first = _chat_system_message(sess.get("lang") or CHAT_LANG_DEFAULT or "it")
    rest = msgs[1:]

    # limita numero di turni utente/assistant
    if len(rest) > CHAT_MAX_MESSAGES:
        rest = rest[-CHAT_MAX_MESSAGES:]

    msgs = [first] + rest

    # limita lunghezza totale in caratteri
    total = sum(len(str(m.get("content", ""))) for m in msgs)
    while total > CHAT_MAX_HISTORY_CHARS and len(msgs) > 2:
        # rimuove il messaggio non-system più vecchio
        msgs.pop(1)
        total = sum(len(str(m.get("content", ""))) for m in msgs)

    sess["messages"] = msgs


def _call_openai_chat(prompt: str, user_id: Optional[str], lang: str = "it") -> str:
    if not CHAT_OPENAI_KEY:
        raise HTTPException(status_code=500, detail="openai_api_key_missing")

    uid, sess = _get_chat_session(user_id, lang)
    _trim_chat_session(sess)

    messages = list(sess.get("messages") or [])
    messages.append({"role": "user", "content": prompt})

    url = f"{CHAT_OPENAI_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {CHAT_OPENAI_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": CHAT_OPENAI_MODEL,
        "messages": messages,
        "max_tokens": CHAT_MAX_TOKENS,
        "user": uid,
    }

    # Evita temperature/top_p non di default sui modelli GPT-5
    if not CHAT_OPENAI_MODEL.startswith("gpt-5"):
        body["temperature"] = CHAT_TEMP
        body["top_p"] = CHAT_TOP_P

    try:
        r = requests.post(url, headers=headers, json=body, timeout=40)
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # vedi punto 4: log migliore
        raise HTTPException(status_code=502, detail=f"openai_request_failed: {e}")


    # aggiorna la history in sessione (user + assistant)
    sess_msgs = list(sess.get("messages") or [])
    sess_msgs.append({"role": "user", "content": prompt})
    sess_msgs.append({"role": "assistant", "content": reply})
    sess["messages"] = sess_msgs

    # applica di nuovo trimming (limite turni / caratteri)
    _trim_chat_session(sess)
    _chat_sessions[uid] = sess

    return reply


_tts_model = None
def get_tts():
    global _tts_model
    if _tts_model is None:
        _tts_model = TTS(model_name=XTTS_MODEL_NAME).to(XTTS_DEVICE)
    return _tts_model

def _speaker_dir(speaker_id: str) -> str:
    p = os.path.join(XTTS_STORE_DIR, speaker_id)
    os.makedirs(p, exist_ok=True)
    return p

# >>> PATCH: cache latents per speaker
def _latents_paths(speaker_id: str) -> Tuple[str, str]:
    d = _speaker_dir(speaker_id)
    return os.path.join(d, "gpt_cond_latents.npy"), os.path.join(d, "speaker_embedding.npy")

def _latents_available(speaker_id: str) -> bool:
    gpt_p, spk_p = _latents_paths(speaker_id)
    return os.path.exists(gpt_p) and os.path.exists(spk_p)

def _compute_and_store_latents(speaker_id: str, refs: List[str]):
    tts = get_tts()
    # molte versioni accettano lista in "get_conditioning_latents"
    gpt_cond_latents, speaker_embedding = tts.get_conditioning_latents(audio_path=refs)
    gp, sp = _latents_paths(speaker_id)
    np.save(gp, gpt_cond_latents)
    np.save(sp, speaker_embedding)

def _load_latents(speaker_id: str):
    gp, sp = _latents_paths(speaker_id)
    return np.load(gp, allow_pickle=True), np.load(sp, allow_pickle=True)

def _list_speaker_refs(speaker_id: str) -> list[str]:
    p = _speaker_dir(speaker_id)
    if not os.path.isdir(p): return []
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = [os.path.join(p, f) for f in os.listdir(p) if os.path.splitext(f)[1].lower() in exts]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[:ENROLL_MAX_FILES]

def _cap_long_silences(wav: np.ndarray, sr: int, max_ms: int = XTTS_MAX_SIL_MS, thr: float = 0.002) -> np.ndarray:
    """
    Comprimi i tratti silenziosi più lunghi di max_ms a max_ms (threshold su ampiezza media mobile).
    Restituisce un array 1D float32. Se max_ms<=0, non modifica l'audio.
    """
    if max_ms is None or max_ms <= 0:
        return np.asarray(wav, dtype=np.float32)

    x = np.asarray(wav, dtype=np.float32)

    # 👇 NUOVO: se per qualche motivo è uno scalare 0-D, non ha senso processarlo
    if x.ndim == 0:
        return x.astype(np.float32)

    if x.ndim == 2:  # stereo -> mono
        x = x.mean(axis=1)
    if x.size == 0:
        return x

    # envelope (media mobile 20 ms)
    win = max(1, int(sr * 0.02))
    kernel = np.ones(win, dtype=np.float32) / win
    env = np.convolve(np.abs(x), kernel, mode="same")
    silent = env < thr

    max_len = int(sr * (max_ms / 1000.0))

    out_segments = []
    n = x.shape[0]
    i = 0
    while i < n:
        j = i
        if silent[i]:
            while j < n and silent[j]:
                j += 1
            run = j - i
            keep = min(run, max_len)
            if keep > 0:
                out_segments.append(np.zeros(keep, dtype=np.float32))
            # se keep==0, rimuove completamente il silenzio lunghissimo
        else:
            while j < n and not silent[j]:
                j += 1
            out_segments.append(x[i:j])  # sempre 1D
        i = j

    y = np.concatenate(out_segments) if out_segments else x

    # normalizzazione soft per headroom
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = 0.95 * (y / peak)
    return y

# >>> NEW: split testo in pezzi brevi per TTS
def _split_tts_text(text: str, max_chars: int = XTTS_CHUNK_CHARS) -> List[str]:
    """
    Divide il testo in spezzoni <= max_chars, preferendo i punti di punteggiatura.
    Serve per generare più clip XTTS e poi unirle.
    """
    txt = re.sub(r"\s+", " ", (text or "")).strip()
    if not txt:
        return []
    if len(txt) <= max_chars:
        return [txt]

    # split grezzo su punteggiatura forte
    parts = re.split(r"(?<=[\.\!\?\:\;])\s+", txt)
    segments: List[str] = []
    cur = ""

    for p in parts:
        if not p:
            continue
        candidate = (cur + " " + p).strip() if cur else p
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                segments.append(cur.strip())
                cur = ""
            # se il pezzo è ancora troppo lungo, spezza per parole
            if len(p) <= max_chars:
                cur = p
            else:
                words = p.split(" ")
                buf = ""
                for w in words:
                    cand = (buf + " " + w).strip() if buf else w
                    if len(cand) <= max_chars:
                        buf = cand
                    else:
                        if buf:
                            segments.append(buf.strip())
                        buf = w
                if buf:
                    cur = buf
    if cur:
        segments.append(cur.strip())
    return segments


# >>> NEW: unione spezzoni audio con cross-fade per evitare click tra i pezzi
def _concat_with_crossfade(chunks: List[np.ndarray], sr: int, cross_ms: int = XTTS_CHUNK_XFADE_MS) -> np.ndarray:
    """
    Concatena più clip 1D float32 aggiungendo un piccolo cross-fade (default 40ms)
    tra un pezzo e l'altro per rendere la voce continua.
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1 or cross_ms <= 0:
        return np.asarray(chunks[0], dtype=np.float32)

    out = np.asarray(chunks[0], dtype=np.float32)
    cross = int(sr * (cross_ms / 1000.0))

    for c in chunks[1:]:
        c = np.asarray(c, dtype=np.float32)
        if cross <= 0:
            out = np.concatenate([out, c])
            continue

        # cross-fade limitato a metà di ciascun chunk per sicurezza
        cf = min(cross, c.shape[0] // 2, out.shape[0] // 2)
        if cf <= 0:
            out = np.concatenate([out, c])
            continue

        fade = np.linspace(0.0, 1.0, cf, dtype=np.float32)
        out[-cf:] = out[-cf:] * (1.0 - fade) + c[:cf] * fade
        out = np.concatenate([out, c[cf:]])

    return out


def _ffmpeg_trim_mono_24k_10s(src: str, dst: str, max_sec: int = ENROLL_MAX_PER_FILE_SEC):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = [
        "ffmpeg","-y","-i", src,
        "-t", str(max_sec),
        "-ar","24000","-ac","1","-c:a","pcm_s16le",
        dst
    ]
    subprocess.run(cmd, check=True)

def _probe_duration_sec(path: str) -> Optional[float]:
    try:
        out = subprocess.run(
            ["ffprobe","-v","error","-select_streams","a:0","-show_entries","format=duration",
             "-of","json", path],
            capture_output=True, text=True, check=True)
        data = json.loads(out.stdout or "{}")
        dur = (data.get("format") or {}).get("duration")
        return float(dur) if dur is not None else None
    except Exception:
        return None

def _json_meta_path(speaker_id: str) -> str:
    return os.path.join(_speaker_dir(speaker_id), "meta.json")

def _save_meta(speaker_id: str, meta: Dict):
    with open(_json_meta_path(speaker_id), "w") as f:
        json.dump(meta, f)

def _load_meta(speaker_id: str) -> Dict:
    p = _json_meta_path(speaker_id)
    if not os.path.exists(p): return {}
    with open(p, "r") as f:
        return json.load(f)

async def _save_many(files: List[UploadFile]) -> List[str]:
    paths = []
    for f in files:
        suffix = os.path.splitext(f.filename or "ref.wav")[1] or ".wav"
        fd, path = tempfile.mkstemp(prefix="xtts_ref_", suffix=suffix)
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await f.read(1024*1024)
                if not chunk: break
                out.write(chunk)
        await f.close()
        paths.append(path)
    return paths

def download_audio_resource(url: str, dest_path: str):
    """Scarica audio da YouTube o URL diretto convertendolo in WAV."""
    
    # CASO 1: YouTube
    if "youtube.com" in url or "youtu.be" in url:
        print(f"[DL] Rilevato YouTube: {url}")
        
        # yt-dlp aggiunge automaticamente l'estensione, quindi passiamo il path senza .wav
        base_path = os.path.splitext(dest_path)[0]
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': base_path, 
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'noplaylist': True,
            'overwrites': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        # Verifica se yt-dlp ha salvato come nome.wav
        final_file = base_path + ".wav"
        if os.path.exists(final_file) and final_file != dest_path:
            shutil.move(final_file, dest_path)
        return

    # CASO 2: File Diretto (MP3/WAV su server)
    print(f"[DL] Download diretto: {url}")
    r = requests.get(url, stream=True, timeout=30)
    if not r.ok:
        raise Exception(f"Errore download URL: {r.status_code}")
    with open(dest_path, "wb") as out:
        for chunk in r.iter_content(1024*1024):
            if chunk: out.write(chunk)

app.mount("/static/audio", StaticFiles(directory=PUBLIC_AUDIO_DIR), name="audio")
# =========================
# ======== XTTS API =======
# =========================

@app.post("/v1/xtts/enroll_urls")
async def xtts_enroll_urls(
    speaker_id: str = Form(...),
    ref_url: List[str] = Form(...),
    authorization: Optional[str] = Header(None),):
    _require_auth(authorization)
    if not ref_url:
        raise HTTPException(400, "missing_ref_urls")

    sp_dir = _speaker_dir(speaker_id)
    saved = []
    
    for i, url in enumerate(ref_url[:ENROLL_MAX_FILES]):
        # Definiamo il percorso PRIMA per poterlo usare in entrambi i casi
        raw_path = os.path.join(sp_dir, f"raw_{uuid.uuid4().hex}.wav")

        # --- CASO 1: YOUTUBE (Nuova aggiunta) ---
        if "youtube.com" in url or "youtu.be" in url:
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.splitext(raw_path)[0], # yt-dlp gestisce l'estensione
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'quiet': True,
                    'noplaylist': True,
                    'overwrites': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Se yt-dlp ha salvato come nome.wav, rinominiamo se serve
                expected_out = os.path.splitext(raw_path)[0] + ".wav"
                if os.path.exists(expected_out) and expected_out != raw_path:
                    shutil.move(expected_out, raw_path)
            except Exception as e:
                print(f"⚠️ Errore download YouTube {url}: {e}")
                if os.path.exists(raw_path): os.remove(raw_path)
                continue

        # --- CASO 2: FILE NORMALE (Codice originale intoccato) ---
        else:
            r = requests.get(url, stream=True, timeout=30)
            if not r.ok:
                continue
            # raw_path è già definito sopra
            with open(raw_path, "wb") as out:
                for chunk in r.iter_content(1024*1024):
                    if chunk: out.write(chunk)

        # --- PROCESSO COMUNE (Codice originale intoccato) ---
        try:
            dur = _probe_duration_sec(raw_path)
            if dur is not None and dur > ENROLL_MAX_FILE_DURATION_SEC:
                try: os.remove(raw_path)
                except: pass
                continue

            segments = _ffmpeg_split_mono_24k(raw_path, sp_dir, ENROLL_MAX_PER_FILE_SEC)
            room = ENROLL_MAX_FILES - len(saved)
            saved.extend(segments[:max(0, room)])
        finally:
            try: os.remove(raw_path)
            except: pass

    if not saved:
      raise HTTPException(400, "no_refs_saved")

    # Pre-computa e salva latents (clone più fedele)
    try:
        _compute_and_store_latents(speaker_id, _list_speaker_refs(speaker_id))
    except Exception:
        pass

    # Calcola e salva anche il centroid ECAPA per il match
    try:
        _compute_and_store_centroid(speaker_id)
    except Exception:
        pass


    meta = _load_meta(speaker_id) or {}
    meta["speaker_id"] = speaker_id
    meta["refs"] = [os.path.basename(p) for p in _list_speaker_refs(speaker_id)]
    now_iso = __import__("datetime").datetime.utcnow().isoformat()
    meta["updated_at"] = now_iso
    meta.setdefault("created_at", now_iso)
    meta["latents_cached"] = _latents_available(speaker_id)
    _save_meta(speaker_id, meta)

    return {"ok": True, "speaker_id": speaker_id, "refs": meta["refs"], "count": len(meta["refs"]), "latents": bool(meta["latents_cached"])}
# === NEW: ECAPA centroid caching per speaker =================================
def _centroid_path(speaker_id: str) -> str:
    return os.path.join(_speaker_dir(speaker_id), "ecapa_centroid.npy")

def _centroid_available(speaker_id: str) -> bool:
    return os.path.exists(_centroid_path(speaker_id))

def _compute_and_store_centroid(speaker_id: str):
    refs = _list_speaker_refs(speaker_id)
    c = speaker_centroid_from_paths(refs)
    os.makedirs(_speaker_dir(speaker_id), exist_ok=True)
    np.save(_centroid_path(speaker_id), c)

def _load_centroid(speaker_id: str) -> np.ndarray | None:
    p = _centroid_path(speaker_id)
    if not os.path.exists(p): return None
    c = np.load(p, allow_pickle=True)
    c = np.asarray(c, dtype=np.float32)
    if c.shape != (192,): return None
    return c / (np.linalg.norm(c) + 1e-9)

@app.get("/v1/xtts/speakers/{speaker_id}")
async def xtts_speaker_get(speaker_id: str, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    refs = _list_speaker_refs(speaker_id)
    if not refs:
        raise HTTPException(404, "speaker_not_found")
    meta = _load_meta(speaker_id)
    return {"ok": True, "speaker_id": speaker_id, "refs": [os.path.basename(x) for x in refs], "meta": meta}

@app.delete("/v1/xtts/speakers/{speaker_id}")
async def xtts_speaker_delete(speaker_id: str, authorization: Optional[str] = Header(None)):
    _require_auth(authorization)
    p = _speaker_dir(speaker_id)
    if not os.path.isdir(p):
        return {"ok": True, "deleted": False}
    shutil.rmtree(p, ignore_errors=True)
    return {"ok": True, "deleted": True}

@app.post("/v1/xtts/clone")
async def xtts_clone(
    ref: Optional[List[UploadFile]] = File(None),
    speaker_id: Optional[str] = Form(None),
    text: str = Form(...),
    language: Optional[str] = Form("it"),
    format: Optional[str] = Form("wav"),
    sample_rate: Optional[int] = Form(None),
    authorization: Optional[str] = Header(None),
    background_tasks: BackgroundTasks = None,
):
    _require_auth(authorization)
    local_refs: List[str] = []

    try:
        # --- 1) Carica i riferimenti voce ---
        if speaker_id:
            local_refs = _list_speaker_refs(speaker_id)
            if not local_refs:
                raise HTTPException(status_code=404, detail="speaker_refs_not_found")

            # opzionale ma utile: filtra e logga eventuali file non decodificabili
            try:
                bad = _debug_check_refs(local_refs)
                if bad:
                    print(f"[XTTS_CLONE] BAD REFS for {speaker_id}: {bad}")
                    local_refs = [p for p in local_refs if p not in bad]
                    if not local_refs:
                        raise HTTPException(status_code=500, detail="speaker_refs_invalid")
            except Exception as e:
                # se torchaudio o altro va in errore, non bloccare la request
                print(f"[XTTS_CLONE] _debug_check_refs failed: {e}")
        else:
            if not ref:
                raise HTTPException(status_code=400, detail="missing_ref_or_speaker_id")
            local_refs = await _save_many(ref)

        # --- 2) Prepara testo e modello TTS ---
        tts = get_tts()
        lang = (language or "it").lower()
        say = normalize_text_it(text) if lang.startswith("it") else text

        # --- 3) Genera audio (latents -> fallback speaker_wav) ---
        async with tts_sem:
            wav = None

            # 3a) prova con latents se disponibili
            if speaker_id and _latents_available(speaker_id):
                try:
                    gpt_lat, spk_emb = _load_latents(speaker_id)
                    try:
                        wav = tts.tts(
                            text=say,
                            language=language or "it",
                            gpt_cond_latents=gpt_lat,
                            speaker_embedding=spk_emb,
                            split_sentences=XTTS_SPLIT_SENTENCES,
                        )
                    except TypeError:
                        if hasattr(tts, "tts_with_preset"):
                            wav = tts.tts_with_preset(
                                text=say,
                                language=language or "it",
                                gpt_cond_latents=gpt_lat,
                                speaker_embedding=spk_emb,
                                split_sentences=XTTS_SPLIT_SENTENCES,
                            )
                except Exception as e:
                    print(f"[XTTS_CLONE] latents path failed for {speaker_id}: {e}")
                    wav = None

            # 3b) SEMPRE fallback su speaker_wav se wav è ancora None
            if wav is None:
                try:
                    wav = tts.tts(
                        text=say,
                        speaker_wav=local_refs,
                        language=language or "it",
                        split_sentences=XTTS_SPLIT_SENTENCES,
                    )
                except TypeError:
                    # vecchie versioni senza split_sentences
                    wav = tts.tts(text=say, speaker_wav=local_refs, language=language or "it")
                except Exception as e:
                    print(f"[XTTS_CLONE] speaker_wav error: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"xtts_clone_tts_failed: {e}",
                    )

        # --- 4) Validazione array audio ---
        if wav is None:
            raise HTTPException(status_code=500, detail="xtts_clone_tts_returned_none")

        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)  # mono soft

        if wav.ndim == 0 or wav.size == 0:
            raise HTTPException(status_code=500, detail="xtts_clone_tts_returned_empty")

        # --- 5) Resample + cap silenzi ---
        sr = int(sample_rate or XTTS_SR)
        if sr != XTTS_SR:
            wav = librosa.resample(wav, orig_sr=XTTS_SR, target_sr=sr)

        wav = _cap_long_silences(wav, sr, max_ms=XTTS_MAX_SIL_MS)

        # --- 6) Output file (come prima) ---
        fmt = (format or "wav").lower()
        tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
        tmp.close()

        if fmt == "mp3":
            pcm_path = tmp.name.replace(".mp3", ".wav")
            sf.write(pcm_path, wav, sr, subtype="PCM_16")
            subprocess.run(["ffmpeg", "-y", "-i", pcm_path, tmp.name], check=True)
            try:
                os.remove(pcm_path)
            except Exception:
                pass
            mime = "audio/mpeg"
        else:
            sf.write(tmp.name, wav, sr, subtype="PCM_16")
            mime = "audio/wav"

        if background_tasks is not None:
            background_tasks.add_task(os.remove, tmp.name)

        return FileResponse(tmp.name, media_type=mime)

    finally:
        # se non usi speaker_id, pulisco i wav temporanei caricati
        if not speaker_id and local_refs:
            for p in local_refs:
                try:
                    os.remove(p)
                except Exception:
                    pass

@app.post("/v1/xtts/chat")
async def xtts_chat(
    prompt: str = Form(...),                   # testo da mandare a ChatGPT
    user_id: Optional[str] = Form(None),       # ID univoco utente (passato a OpenAI)
    ref: Optional[List[UploadFile]] = File(None),
    speaker_id: Optional[str] = Form(None),
    language: Optional[str] = Form("it"),
    format: Optional[str] = Form("wav"),
    sample_rate: Optional[int] = Form(None),
    authorization: Optional[str] = Header(None),
    background_tasks: BackgroundTasks = None,
):
    """
    1) Manda 'prompt' a ChatGPT (OpenAI) usando user_id.
    2) Prende la risposta testuale.
    3) La converte in audio XTTS, usando voce clonata (speaker_id o ref),
       ma in PEZZI piccoli che poi unisce con cross-fade per non perdere qualità.
    """
    _require_auth(authorization)

    # ---- gestisci riferimenti voce (come in xtts_clone) ----
    local_refs: List[str] = []
    try:
        if speaker_id:
            local_refs = _list_speaker_refs(speaker_id)
            if not local_refs:
                raise HTTPException(404, "speaker_refs_not_found")

            # DEBUG + filtro ref corrotti
            try:
                bad = _debug_check_refs(local_refs)
                if bad:
                    print(f"[XTTS_CHAT] BAD REFS for {speaker_id}: {bad}")
                    # rimuovi i file che non si decodificano
                    local_refs = [p for p in local_refs if p not in bad]
                    if not local_refs:
                        # tutti i ref di questo speaker sono inutilizzabili
                        raise HTTPException(status_code=500, detail="speaker_refs_invalid")
            except Exception as e:
                # non bloccare la richiesta se il debug fallisce
                print(f"[XTTS_CHAT] _debug_check_refs failed: {e}")
        else:
            if not ref:
                raise HTTPException(400, "missing_ref_or_speaker_id")
            local_refs = await _save_many(ref)

        lang = (language or "it").lower()

        # ---- 1) chiamata ChatGPT ----
        reply_text = _call_openai_chat(prompt=prompt, user_id=user_id, lang=lang)
        if not reply_text:
            raise HTTPException(500, "empty_chatgpt_reply")

        # ---- 2) normalizza testo per TTS ----
        say = normalize_text_it(reply_text) if lang.startswith("it") else reply_text

        # split del testo in spezzoni più piccoli
        segments = _split_tts_text(say, XTTS_CHUNK_CHARS)
        if not segments:
            raise HTTPException(500, "empty_tts_segments")

        tts = get_tts()

        # ---- 3) TTS in spezzoni con stessa voce ----
        async with tts_sem:
            gpt_lat = spk_emb = None
            use_latents = False

            # se esistono latents per lo speaker, li carico UNA volta sola
            if speaker_id and _latents_available(speaker_id):
                try:
                    gpt_lat, spk_emb = _load_latents(speaker_id)
                    use_latents = True
                except Exception:
                    use_latents = False

            chunk_wavs: List[np.ndarray] = []

            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue

                wav_seg = None

                # preferisci latents (più stabili / veloci)
                if use_latents:
                    try:
                        wav_seg = tts.tts(
                            text=seg,
                            language=language or "it",
                            gpt_cond_latents=gpt_lat,
                            speaker_embedding=spk_emb,
                            split_sentences=False,   # facciamo noi lo split
                        )
                    except TypeError:
                        if hasattr(tts, "tts_with_preset"):
                            wav_seg = tts.tts_with_preset(
                                text=seg,
                                language=language or "it",
                                gpt_cond_latents=gpt_lat,
                                speaker_embedding=spk_emb,
                                split_sentences=False,
                            )
                        else:
                            wav_seg = None
                    except Exception:
                        wav_seg = None

                # fallback: lista di wav di riferimento
                if wav_seg is None:
                    try:
                        wav_seg = tts.tts(
                            text=seg,
                            speaker_wav=local_refs,
                            language=language or "it",
                            split_sentences=False,
                        )
                    except TypeError:
                        wav_seg = tts.tts(text=seg, speaker_wav=local_refs, language=language or "it")
                    except Exception as e:
                        # LOG UTILE
                        print(f"[XTTS_CHAT] speaker_wav error for segment='{seg[:80]}': {e}")
                        raise HTTPException(status_code=500, detail=f"xtts_tts_failed: {e}")

                wav_seg = np.asarray(wav_seg, dtype=np.float32).flatten()
                if wav_seg.size > 0:
                    chunk_wavs.append(wav_seg)

            if not chunk_wavs:
                raise HTTPException(500, "tts_failed")

        # unisci tutti i pezzi con cross-fade
        wav = _concat_with_crossfade(chunk_wavs, XTTS_SR, cross_ms=XTTS_CHUNK_XFADE_MS)

        # resample + cap silenzi come nel clone normale
        wav = np.asarray(wav, dtype=np.float32)
        sr = int(sample_rate or XTTS_SR)
        if wav.ndim == 0 or wav.size == 0:
            raise HTTPException(status_code=500, detail="xtts_clone_tts_returned_empty")

        if sr != XTTS_SR:
            wav = librosa.resample(wav, orig_sr=XTTS_SR, target_sr=sr)

        wav = _cap_long_silences(wav, sr, max_ms=XTTS_MAX_SIL_MS)

        # ---- output audio (uguale a /v1/xtts/clone) ----
        fmt = (format or "wav").lower()
        tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
        tmp.close()

        if fmt == "mp3":
            pcm_path = tmp.name.replace(".mp3", ".wav")
            sf.write(pcm_path, wav, sr, subtype="PCM_16")
            subprocess.run(["ffmpeg", "-y", "-i", pcm_path, tmp.name], check=True)
            try:
                os.remove(pcm_path)
            except Exception:
                pass
            mime = "audio/mpeg"
        else:
            sf.write(tmp.name, wav, sr, subtype="PCM_16")
            mime = "audio/wav"

        if background_tasks is not None:
            background_tasks.add_task(os.remove, tmp.name)

        return FileResponse(tmp.name, media_type=mime)
    finally:
        # se non usi speaker_id, pulisco i wav temporanei caricati
        if not speaker_id and local_refs:
            for p in local_refs:
                try:
                    os.remove(p)
                except Exception:
                    pass

@app.post("/v1/voicecheck")
async def voicecheck(
    files: List[UploadFile] = File(...),
    authorization: Optional[str] = Header(None)):
    _require_auth(authorization)

    results = []
    try:
        for f in files:
            # salva temporaneo
            import tempfile, os
            suffix = os.path.splitext(f.filename or "audio.wav")[1] or ".wav"
            fd, path = tempfile.mkstemp(prefix="vc_", suffix=suffix)
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            await f.close()

            # analisi (CUDA con SpeechBrain ECAPA + VAD)
            info = analyze_file(path)
            info["filename"] = f.filename
            results.append(info)

            # cleanup
            try: os.remove(path)
            except: pass

        # risposte "semplici"
        if len(results) == 1:
            r = results[0]
            return {
                "ok": True,
                "speakers": r["speakers"],
                "multi_speaker": r["multi_speaker"],
                "dur_sec": r["dur_sec"]
            }
        else:
            # batch
            return {
                "ok": True,
                "files": [
                    {
                        "filename": r.get("filename"),
                        "speakers": r["speakers"],
                        "multi_speaker": r["multi_speaker"],
                        "dur_sec": r["dur_sec"]
                    } for r in results
                ]
            }
    except Exception as e:
        raise HTTPException(500, f"voicecheck_error: {e}")
    

# =========================
# ====== WEBSOCKET ========
# =========================
@app.websocket("/v1/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 1. HANDSHAKE INIZIALE
    # Il client invia un JSON con { token, user_id, language, speaker_id }
    try:
        init_data = await websocket.receive_json()
        token = init_data.get("token")
        # TODO: Validazione token opzionale qui (chiamata a Node auth/me)
        
        user_id = init_data.get("user_id", "anon")
        lang = init_data.get("language", "it")
        
        # MODALITÀ "IMPARO DA TE":
        # Se l'app non manda uno speaker_id (o manda null), usiamo l'user_id.
        # Questo significa che l'AI userà la voce dell'utente stesso (clonazione istantanea).
        speaker_id = init_data.get("speaker_id") or user_id 
        
        print(f"[WS] Connesso User: {user_id} | Speaker Target: {speaker_id}")
        
    except Exception as e:
        print(f"[WS] Handshake error: {e}")
        await websocket.close()
        return

    # Inizializza sessione chat (memoria contesto per ChatGPT)
    _, session = _get_chat_session(user_id, lang)
    audio_buffer = bytearray()
    
    # Genera un ID gruppo univoco per questo scambio (Domanda + Risposta)
    current_group_id = str(uuid.uuid4())

    try:
        while True:
            # Ricezione Messaggi dal Client (App Angular)
            message = await websocket.receive()

            # --- A. STREAMING AUDIO IN INGRESSO ---
            if "bytes" in message:
                audio_buffer.extend(message["bytes"])
                
            # --- B. COMANDI TESTUALI ---
            elif "text" in message:
                text_msg = message["text"]
                
                # Variabile per contenere il testo dell'utente (trascritto o digitato)
                final_user_text = ""
                
                # CASO 1: L'UTENTE HA FINITO DI PARLARE ("END_SPEECH")
                if text_msg == "END_SPEECH":
                    if len(audio_buffer) == 0: continue
                    
                    await websocket.send_json({"status": "processing", "step": "transcribing"})
                    
                    # 1. Salva Audio Utente (Pubblico)
                    user_filename = f"u_{user_id}_{uuid.uuid4().hex[:10]}.wav"
                    user_wav_path = os.path.join(PUBLIC_AUDIO_DIR, user_filename)
                    with open(user_wav_path, "wb") as f:
                        f.write(audio_buffer)
                    
                    # 2. ADDESTRAMENTO ISTANTANEO (Zero-Shot)
                    # Copia l'audio appena ricevuto nella cartella di addestramento dello speaker.
                    # XTTS userà questo file IMMEDIATAMENTE per clonare meglio la voce.
                    spk_dir = os.path.join(XTTS_STORE_DIR, speaker_id)
                    os.makedirs(spk_dir, exist_ok=True)
                    train_ref_path = os.path.join(spk_dir, f"ref_{int(time.time())}.wav")
                    shutil.copy(user_wav_path, train_ref_path)
                    
                    # 3. Trascrizione (ASR Whisper)
                    # Usa float16 se sei su GPU per massima velocità
                    model = get_model(DEFAULT_CTYPE) 
                    segments, _ = model.transcribe(user_wav_path, language=lang, beam_size=2)
                    final_user_text = " ".join([s.text for s in segments]).strip()
                    
                    if not final_user_text:
                        await websocket.send_json({"status": "error", "msg": "Non ho sentito nulla"})
                        audio_buffer = bytearray()
                        continue

                    # Notifica all'app il testo capito
                    await websocket.send_json({
                        "status": "transcription", 
                        "text": final_user_text, 
                        "groupId": current_group_id
                    })
                    
                    # Reset buffer audio per il prossimo turno
                    audio_buffer = bytearray()

                # CASO 2: INPUT TESTUALE DIRETTO ("TEXT_INPUT:...")
                elif text_msg.startswith("TEXT_INPUT:"):
                    final_user_text = text_msg.replace("TEXT_INPUT:", "").strip()
                    if not final_user_text: continue
                    
                    # Feedback immediato
                    await websocket.send_json({
                        "status": "transcription", 
                        "text": final_user_text, 
                        "groupId": current_group_id
                    })
                    # (Qui non c'è audio utente da salvare o addestrare)

                # --- SE ABBIAMO UN TESTO UTENTE, PROCEDIAMO CON LA RISPOSTA AI ---
                if final_user_text:
                    
                    # 4. Generazione Risposta (LLM ChatGPT)
                    # Istruzione per imitare lo stile
                    style_instruction = (
                        f"L'utente ha detto: '{final_user_text}'. "
                        "Rispondi brevemente. Imita il tono e lo stile dell'utente (es. se è formale sii formale, se scherza scherza)."
                    )
                    reply_text = _call_openai_chat(prompt=style_instruction, user_id=user_id, lang=lang)
                    
                    await websocket.send_json({
                        "status": "reply_text", 
                        "text": reply_text, 
                        "groupId": current_group_id
                    })
                    
                    # 5. Generazione Audio (TTS XTTS)
                    await websocket.send_json({"status": "processing", "step": "generating_audio"})
                    
                    # Recupera i file di riferimento (incluso quello appena salvato se era vocale!)
                    ref_wavs = _list_speaker_refs(speaker_id)
                    # Fallback se non ci sono ref (es. primo messaggio testuale su nuovo speaker)
                    if not ref_wavs and 'user_wav_path' in locals(): ref_wavs = [user_wav_path]
                    
                    tts = get_tts()
                    # Splitta in frasi per streaming fluido
                    sentences = _split_tts_text(reply_text, max_chars=200)
                    
                    full_ai_audio = [] # Accumulatore per il file finale
                    
                    for sent in sentences:
                        if not sent.strip(): continue
                        
                        try:
                            # Genera audio clonando la voce dai ref_wavs
                            out_wav = tts.tts(
                                text=sent, 
                                language=lang,
                                speaker_wav=ref_wavs, # <--- CLONAZIONE DINAMICA
                                split_sentences=False
                            )
                            
                            # Conversione Float32 -> Int16 per streaming
                            wav_np = np.array(out_wav, dtype=np.float32)
                            full_ai_audio.extend(wav_np)
                            
                            wav_np_norm = wav_np / (np.max(np.abs(wav_np)) + 1e-9)
                            wav_int16 = (wav_np_norm * 32767).astype(np.int16)
                            
                            # Invia chunk audio al client
                            await websocket.send_bytes(wav_int16.tobytes())
                        except Exception as tts_err:
                            print(f"[WS] TTS Error on chunk: {tts_err}")

                    # 6. Salvataggio e Sync (Node.js)
                    
                    # Salva file AI completo su disco pubblico
                    ai_filename = f"ai_{user_id}_{uuid.uuid4().hex[:10]}.wav"
                    ai_wav_path = os.path.join(PUBLIC_AUDIO_DIR, ai_filename)
                    # XTTS_SR è definito in app.py (di solito 24000)
                    sf.write(ai_wav_path, full_ai_audio, XTTS_SR)
                    
                    # Costruisci URL
                    # Sostituisci con il tuo dominio pubblico reale in produzione
                    base_url = "https://ml.nexipse.it/static/audio" 
                    
                    # Prepara payload per Node
                    payload = {
                        "userId": user_id,
                        "groupId": current_group_id,
                        "userText": final_user_text,
                        # Se l'input era testo, non c'è userAudioUrl
                        "userAudioUrl": f"{base_url}/{user_filename}" if 'user_filename' in locals() else None,
                        "aiText": reply_text,
                        "aiAudioUrl": f"{base_url}/{ai_filename}",
                        "speakerId": speaker_id,
                        "voiceId": speaker_id # Per compatibilità col modello
                    }
                    
                    # Chiamata Sync a Node (Fire & Forget con timeout breve)
                    try:
                        print(f"[WS] Syncing to Node: {NODE_API_URL}/internal/save-chat")
                        requests.post(f"{NODE_API_URL}/internal/save-chat", json=payload, timeout=5)
                    except Exception as err:
                        print(f"[WS] ERRORE SYNC NODE: {err}")

                    # 7. Fine Turno
                    await websocket.send_json({"status": "done", "groupId": current_group_id})
                    
                    # Reset variabili loop
                    current_group_id = str(uuid.uuid4())
                    if 'user_filename' in locals(): del user_filename
                    if 'user_wav_path' in locals(): del user_wav_path

    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {user_id}")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try: await websocket.close()
        except: pass# =========================
# ======== ASR API ========
# =========================

# cache modelli per ctype
_MODEL_CACHE: Dict[str, WhisperModel] = {}

def get_model(compute_type: str):
    m = _MODEL_CACHE.get(compute_type)
    if m is None:
        m = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=compute_type)
        _MODEL_CACHE[compute_type] = m
    return m

class Segment(BaseModel):
    start: float
    end: float
    text: str

class TranscriptionResponse(BaseModel):
    ok: bool = True
    model: str
    id: str
    language: Optional[str]
    duration: Optional[float]
    text: str
    segments: List[Segment]

# --- vicino agli altri Pydantic models ---
class SummarizeReq(BaseModel):
    text: Optional[str] = None
    segments: Optional[List[Segment]] = None  # riuso il tuo Segment (start,end,text)
    lang: Optional[str] = "it"
    style: Optional[str] = "paragraph"        # "paragraph" | "bullets"
    words: Optional[int] = 180

class SummarizeResp(BaseModel):
    ok: bool = True
    model: str
    summary: str
    bullets: Optional[List[str]] = None

def _join_segments(segs: Optional[List[Segment]]) -> str:
    if not segs: return ""
    # concat semplice con spazio (evita raddoppi punteggiatura)
    return " ".join((s.text or "").strip() for s in segs if (s.text or "").strip()).strip()


def _pick_params(duration: Optional[float], quality: str) -> Tuple[bool,int,int,int,dict,bool]:
    q = (quality or "").lower().strip()
    if q == "fast":
        vad_filter = VAD_FILTER
        beam_size, best_of, chunk_length = 1, 1, 30
        vad_params = dict(min_silence_duration_ms=400, speech_pad_ms=200)   # <-- int
        cond_prev = False
    elif q == "accurate":
        vad_filter = VAD_FILTER
        beam_size, best_of, chunk_length = 5, 5, 60
        vad_params = dict(min_silence_duration_ms=500, speech_pad_ms=300)   # <-- int
        cond_prev = True
    else:
        if duration is None:
            vad_filter = VAD_FILTER
            beam_size, best_of, chunk_length = 5, 5, 60
            vad_params = dict(min_silence_duration_ms=500, speech_pad_ms=300)   # <-- int
            cond_prev = True
        elif duration <= 15:
            vad_filter = False
            beam_size, best_of, chunk_length = 5, 5, 30
            vad_params = {}
            cond_prev = True
        elif duration <= 3600:
            vad_filter = VAD_FILTER
            beam_size, best_of, chunk_length = 5, 5, 60
            vad_params = dict(min_silence_duration_ms=500, speech_pad_ms=300)   # <-- int
            cond_prev = True
        elif duration <= 4*3600:
            vad_filter = VAD_FILTER
            beam_size, best_of, chunk_length = 3, 3, 90
            vad_params = dict(min_silence_duration_ms=600, speech_pad_ms=400)   # <-- int
            cond_prev = True
        else:
            vad_filter = VAD_FILTER
            beam_size, best_of, chunk_length = 2, 2, 120
            vad_params = dict(min_silence_duration_ms=800, speech_pad_ms=500)   # <-- int
            cond_prev = True

    if FORCE_CHUNK > 0:
        chunk_length = FORCE_CHUNK

    return (vad_filter, beam_size, best_of, chunk_length, vad_params, cond_prev)

os.environ.setdefault("TORCH_DISABLE_CUDA", "0")

@app.get("/v1/health")
async def health():
    try:
        get_model(DEFAULT_CTYPE)              # warmup minimo ASR
        if os.getenv("WARM_TTS","false").lower() == "true":
            get_tts()                         # solo se vuoi davvero scaldarlo
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

async def _save_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "audio")[1]
    fd, path = tempfile.mkstemp(prefix="ml_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk: break
            f.write(chunk)
    await upload.close()
    return path

@app.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    translate: Optional[bool] = Form(False),
    quality: Optional[str] = Form(None),             # "fast" | "balanced" | "accurate"
    compute_type: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
    x_quality: Optional[str] = Header(None)):
    _require_auth(authorization)

    tmp_path = await _save_temp(file)

    # limite dimensione file
    try:
        st = os.stat(tmp_path)
        if MAX_FILE_MB > 0 and st.st_size > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail="file_too_large")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="temp_file_missing")

    try:
        dur = _probe_duration_sec(tmp_path)
        q = (quality or x_quality or DEFAULT_QUALITY or "balanced")
        vad_filter, beam_size, best_of, chunk_length, vad_params, cond_prev = _pick_params(dur, q)

        ctype = (compute_type or DEFAULT_CTYPE).strip()
        model = get_model(ctype)

        async with sem:
            try:
                segments_iter, info = model.transcribe(
                    tmp_path,
                    language=language or DEFAULT_LANGUAGE,
                    task="translate" if translate else "transcribe",
                    vad_filter=vad_filter,
                    vad_parameters=vad_params or None,
                    beam_size=beam_size,
                    best_of=best_of,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=cond_prev,
                    chunk_length=chunk_length,
                    initial_prompt=INITIAL_PROMPT,
                )
            except Exception as e:
                # fallback compute type su OOM
                s = str(e).lower()
                if ("cuda out of memory" in s or "cublas" in s or "allocator" in s) and ctype != FALLBACK_CTYPE:
                    model = get_model(FALLBACK_CTYPE)
                    segments_iter, info = model.transcribe(
                        tmp_path,
                        language=language or DEFAULT_LANGUAGE,
                        task="translate" if translate else "transcribe",
                        vad_filter=vad_filter,
                        vad_parameters=vad_params or None,
                        beam_size=max(1, beam_size-1),
                        best_of=max(1, best_of-1),
                        temperature=0.0,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=cond_prev,
                        chunk_length=min(120, chunk_length + 30),
                        initial_prompt=INITIAL_PROMPT,
                    )
                else:
                    raise

            segs: List[Segment] = []
            parts: List[str] = []
            for s in segments_iter:
                text = s.text
                segs.append(Segment(start=float(s.start), end=float(s.end), text=text))
                parts.append(text)

            return {
                "ok": True,
                "model": f"faster-whisper:{MODEL_SIZE}:{ctype}",
                "id": str(uuid.uuid4()),
                "language": getattr(info, "language", None),
                "duration": float(getattr(info, "duration", 0.0)) if getattr(info, "duration", None) is not None else None,
                "text": (" ".join(parts)).strip(),
                "segments": [s.model_dump() for s in segs],
            }
    finally:
        try: os.remove(tmp_path)
        except Exception: pass


@app.post("/v1/voice/grade")
async def voice_grade(
    file: UploadFile = File(...),
    speaker_id: Optional[str] = Form(None),
    is_tts: Optional[str] = Form(None), 
    authorization: Optional[str] = Header(None),
):
    _require_auth(authorization)
    tmp_path = await _save_temp(file)
    try:
        centroid = _load_centroid(speaker_id) if speaker_id else None
        bank = None
        if speaker_id:
            refs = _list_speaker_refs(speaker_id)
            if refs:
                bank = build_profile_bank({speaker_id: refs})
        tts_flag = str(is_tts).lower() in ("1", "true", "yes", "on")
        out = grade_clip(
            path=tmp_path,
            profile_centroid=centroid,
            profile_bank=bank,
            is_tts=tts_flag,
        )
        return out
    finally:
        try: os.remove(tmp_path)
        except: pass

@app.post("/v1/voice/rebuild_profile")
async def voice_rebuild_profile(
    speaker_id: str = Form(...),
    authorization: Optional[str] = Header(None),
):
    _require_auth(authorization)
    _compute_and_store_centroid(speaker_id)
    meta = _load_meta(speaker_id) or {}
    meta["ecapa_centroid"] = True
    _save_meta(speaker_id, meta)
    return {"ok": True, "speaker_id": speaker_id, "centroid": True}
@app.post("/v1/summarize", response_model=SummarizeResp)
async def summarize_api(
    req: SummarizeReq,
    authorization: Optional[str] = Header(None),
):
    _require_auth(authorization)
    # prepara testo
    body_text = (req.text or "").strip()
    if not body_text and req.segments:
        body_text = _join_segments(req.segments)
    if not body_text:
        raise HTTPException(400, "missing_text_or_segments")

    # chiama backend LLM con semaforo
    async with sum_sem:
        out = _do_summarize(
            text=body_text,
            lang=(req.lang or "it"),
            style=(req.style or "paragraph"),
            max_words=int(req.words or 180),
        )
    return {
        "ok": True,
        "model": out.model,
        "summary": out.summary,
        "bullets": out.bullets,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","9000")), reload=False)
