# voicecheck.py
import math
import importlib
from typing import List, Dict, Any, Tuple

import numpy as np
import soundfile as sf
import librosa
import torch
from sklearn.cluster import AgglomerativeClustering
import tempfile, subprocess, os
from sklearn.metrics import silhouette_score
import logging
# voicecheck.py
import tempfile, subprocess, os
def embed_one(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Ritorna un embedding (1, 192) normalizzato L2 per un chunk audio mono.
    """
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000

    # normalizza ampiezza in ingresso (safety)
    if wav.size:
        peak = float(np.max(np.abs(wav)))
        if peak > 0:
            wav = 0.98 * (wav / peak)

    model = _get_embedder()  # SpeechBrain ECAPA
    wav_t = torch.tensor(wav, dtype=torch.float32, device=_DEVICE).unsqueeze(0)  # [1, T]

    with torch.no_grad():
        # encode_batch restituisce [B, D] o simile -> estrai e porta a numpy
        emb = model.encode_batch(wav_t).squeeze().detach().cpu().numpy().astype(np.float32)

    # forza shape 1x192 e normalizza L2
    emb = emb.reshape(1, -1)
    n = float(np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    emb = emb / n
    return emb  # (1, 192)

def _ffmpeg_decode_to_wav_mono16k(src: str) -> tuple[np.ndarray, int]:
    fd, tmp = tempfile.mkstemp(prefix="dec_", suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg","-hide_banner","-loglevel","error","-nostdin",
             "-y","-i", src, "-vn","-sn","-dn",
             "-ar","16000","-ac","1","-c:a","pcm_s16le", tmp],
            check=True
        )
        wav, sr = sf.read(tmp, always_2d=False)
        return wav, sr
    finally:
        try: os.remove(tmp)
        except: pass

def load_audio_mono_16k(path: str) -> Tuple[np.ndarray, int]:
    try:
        wav, sr = sf.read(path, always_2d=False)
    except Exception:
        # ← fallback universale (risolve l'errore su .m4a)
        wav, sr = _ffmpeg_decode_to_wav_mono16k(path)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        wav = 0.98 * (wav / peak)
    return wav, sr

# -----------------------------
# Globals (lazy init, usa CUDA)
# -----------------------------
_SB_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_webrtcvad = None


def _lazy_webrtcvad():
    """Importa webrtcvad solo al primo uso (evita crash all'avvio)."""
    global _webrtcvad
    if _webrtcvad is None:
        _webrtcvad = importlib.import_module("webrtcvad")
    return _webrtcvad


def _get_embedder():
    """Lazy init di SpeechBrain ECAPA; usa CUDA se disponibile."""
    global _SB_MODEL
    if _SB_MODEL is None:
        EncoderClassifier = importlib.import_module(
            "speechbrain.pretrained"
        ).EncoderClassifier
        _SB_MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": _DEVICE},
        )
    return _SB_MODEL


def _energy_vad_segments_fallback(
    wav: np.ndarray,
    sr: int,
    frame_ms: int,
    min_speech_ms: int,
    max_gap_ms: int,
    env_win_ms: float = 20.0,
) -> List[Tuple[int, int]]:
    """
    Fallback VAD semplice basato su energia (se 'webrtcvad' non è disponibile).
    Non è accurato come WebRTC ma evita di fallire.
    """
    n = len(wav)
    if n == 0:
        return []

    frame_len = max(1, int(sr * frame_ms / 1000))
    hop = frame_len

    # envelope (media mobile)
    win = max(1, int(sr * (env_win_ms / 1000.0)))
    kernel = np.ones(win, dtype=np.float32) / win
    env = np.convolve(np.abs(wav), kernel, mode="same")

    # soglia adattiva: tra media e 85° percentile
    thr = max(float(np.mean(env)) * 2.0, float(np.percentile(env, 85)) * 0.2)

    frames = []
    for i in range(0, n - frame_len + 1, hop):
        e = float(np.mean(env[i : i + frame_len]))
        frames.append((i, i + frame_len, e > thr))

    segs: List[Tuple[int, int]] = []
    cur_start = None
    last_end = None
    max_gap = int(sr * max_gap_ms / 1000)
    min_len = int(sr * min_speech_ms / 1000)

    for (a, b, flag) in frames:
        if flag:
            if cur_start is None:
                cur_start = a
            last_end = b
        else:
            if cur_start is not None and last_end is not None:
                if a - last_end > max_gap:
                    if (last_end - cur_start) >= min_len:
                        segs.append((cur_start, last_end))
                    cur_start, last_end = None, None

    if cur_start is not None and last_end is not None:
        if (last_end - cur_start) >= min_len:
            segs.append((cur_start, last_end))

    return segs


def vad_segments(
    wav: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    aggressiveness: int = 2,
    min_speech_ms: int = 300,
    max_gap_ms: int = 300,
) -> List[Tuple[int, int]]:
    """
    Segmenti [start_sample, end_sample] di parlato.
    - aggressiveness: 0..3 (3 = più severo)
    - min_speech_ms: scarta clip troppo brevi
    - max_gap_ms: unisce sottosegmenti separati da pause brevi
    """
    try:
        webrtcvad = _lazy_webrtcvad()
        vad = webrtcvad.Vad(aggressiveness)
        frame_len = int(sr * frame_ms / 1000)
        hop = frame_len
        n = len(wav)
        frames = []
        for i in range(0, n - frame_len + 1, hop):
            chunk = wav[i : i + frame_len]
            pcm16 = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            is_speech = vad.is_speech(pcm16, sr)
            frames.append((i, i + frame_len, is_speech))

        segs: List[Tuple[int, int]] = []
        cur_start = None
        last_end = None
        max_gap = int(sr * max_gap_ms / 1000)
        min_len = int(sr * min_speech_ms / 1000)

        for (a, b, flag) in frames:
            if flag:
                if cur_start is None:
                    cur_start = a
                last_end = b
            else:
                if cur_start is not None and last_end is not None:
                    if a - last_end > max_gap:
                        if (last_end - cur_start) >= min_len:
                            segs.append((cur_start, last_end))
                        cur_start, last_end = None, None

        if cur_start is not None and last_end is not None:
            if (last_end - cur_start) >= min_len:
                segs.append((cur_start, last_end))
        return segs

    except ImportError:
        # fallback se 'webrtcvad' non è installato
        return _energy_vad_segments_fallback(
            wav, sr, frame_ms, min_speech_ms, max_gap_ms
        )


def cut_chunks(
    wav: np.ndarray, sr: int, segs: List[Tuple[int, int]], max_chunk_sec: float = 5.0
) -> List[np.ndarray]:
    """Spacca i segmenti VAD in sottoclip <= max_chunk_sec per embedding stabili."""
    chunks: List[np.ndarray] = []
    max_len = int(sr * max_chunk_sec)
    for s, e in segs:
        seg = wav[s:e]
        if len(seg) <= max_len:
            chunks.append(seg)
        else:
            for i in range(0, len(seg), max_len):
                part = seg[i : i + max_len]
                if len(part) >= int(sr * 0.8):  # scarta briciole troppo corte
                    chunks.append(part)
    return chunks


EMBED_DIM = 192

def embed_chunks(chunks, sr):
    embs = []
    for i, ch in enumerate(chunks):
        try:
            e = embed_one(ch, sr)   # <-- usa qui la tua funzione reale di embedding
        except Exception as ex:
            logging.exception("embed failed on chunk %d: %s", i, ex)
            continue

        e = np.asarray(e)

        # scarta scalari / liste vuote
        if e.size == 0 or e.ndim == 0:
            logging.warning("empty/scalar embedding on chunk %d -> skip (shape=%s)", i, e.shape)
            continue

        # porta a 2D (1, D) se necessario
        if e.ndim == 1:
            e = e.reshape(1, -1)

        # solo (k, 192) è valido
        if e.shape[1] != EMBED_DIM:
            logging.warning("bad embedding shape %s on chunk %d -> skip", e.shape, i)
            continue

        # tipizza e accumula
        embs.append(e.astype(np.float32, copy=False))

    if not embs:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    return np.concatenate(embs, axis=0)

def estimate_speakers(embs: np.ndarray, distance_threshold: float = 0.35) -> int:
    """
    Stima il numero di parlanti con AgglomerativeClustering su distanza coseno.
    Soglie tipiche: 0.30..0.40 (più basso = merge più aggressivo).
    """
    n = embs.shape[0]
    if n == 0:
        return 0
    if n == 1:
        return 1
    cl = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = cl.fit_predict(embs)
    return int(len(set(labels)))


# -----------------------------
# Public API
# -----------------------------
def analyze_file(path: str) -> Dict[str, Any]:
    """
    Output:
      {
        "ok": True,
        "speakers": <int>,
        "multi_speaker": <bool>,
        "segments": <int>,
        "chunks": <int>,
        "dur_sec": <float>
      }
    """
    wav, sr = load_audio_mono_16k(path)
    dur = float(len(wav)) / sr if sr else 0.0

    # VAD (WebRTC o fallback energy-based)
    segs = vad_segments(
        wav,
        sr,
        frame_ms=30,
        aggressiveness=2,
        min_speech_ms=400,
        max_gap_ms=300,
    )

    # Sottoclip di max 5s per embedding stabile
    chunks = speech_sliding_chunks(wav, sr, segs, win_s=1.5, hop_s=0.75)


    # Embedding + clustering
    embs = embed_chunks(chunks, sr=sr)
    spk_count, spk_conf, *_ = robust_speaker_count(embs)
    return {
        "ok": True,
        "speakers": int(spk_count),
        "multi_speaker": bool(spk_count >= 2),
        "segments": len(segs),
        "chunks": len(chunks),
        "dur_sec": round(dur, 2),
    }

# --- NEW: quality features ----------------------------------------------------
def _snr_db_from_env(env: np.ndarray) -> float:
    # env = envelope (assoluta smussata)
    if env.size == 0:
        return 0.0
    noise = np.percentile(env, 10)
    speech = np.percentile(env, 60)
    noise = float(max(noise, 1e-7))
    speech = float(max(speech, noise + 1e-7))
    return 10.0 * math.log10(speech / noise)

def _clip_ratio(wav: np.ndarray, thr: float = 0.98) -> float:
    if wav.size == 0:
        return 0.0
    return float(np.mean(np.abs(wav) >= thr))

def _envelope(wav: np.ndarray, sr: int, ms: float = 20.0) -> np.ndarray:
    win = max(1, int(sr * (ms / 1000.0)))
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(np.abs(wav.astype(np.float32)), k, mode="same")

def _map_piecewise(x: float, xs: List[float], ys: List[float]) -> float:
    # Interpolazione lineare a tratti clampata
    assert len(xs) == len(ys) and len(xs) >= 2
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1, y0, y1 = xs[i-1], xs[i], ys[i-1], ys[i]
            t = (x - x0) / max(1e-9, (x1 - x0))
            return y0 + t * (y1 - y0)
    return ys[-1]

def _label_from_score(s: float) -> str:
    if s < 40: return "Debole"
    if s < 60: return "Discreto"
    if s < 75: return "Buono"
    if s < 90: return "Molto buono"
    return "Eccellente"

def _pairwise_cosine(embs: np.ndarray) -> Tuple[float, float]:
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim != 2 or embs.shape[0] == 0:
        return (1.0, 1.0)
    # safety: normalizza comunque
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    n = embs.shape[0]
    if n == 1:
        return (1.0, 1.0)

    sims = []
    for i in range(n):
        vi = embs[i]
        for j in range(i + 1, n):
            vj = embs[j]
            sims.append(float(np.dot(vi, vj)))  # cosine perché già L2-normalizzati
    sims = np.array(sims, dtype=np.float32)
    return float(np.mean(sims)), float(np.percentile(sims, 5))


def speaker_centroid_from_paths(paths: List[str]) -> np.ndarray:
    # carica audio → VAD → chunk → embedding → centroid normalizzato
    all_chunks = []
    for p in paths:
        try:
            wav, sr = load_audio_mono_16k(p)
            segs = vad_segments(wav, sr, frame_ms=30, aggressiveness=2, min_speech_ms=300, max_gap_ms=300)
            chunks = cut_chunks(wav, sr, segs, max_chunk_sec=5.0)
            all_chunks.extend(chunks)
        except Exception:
            pass
    if not all_chunks:
        return np.zeros((192,), dtype=np.float32)
    embs = embed_chunks(all_chunks, sr=16000)
    if embs.shape[0] == 0:
        return np.zeros((192,), dtype=np.float32)
    c = np.mean(embs, axis=0)
    c /= (np.linalg.norm(c) + 1e-9)
    return c.astype(np.float32)

# === helper nuovi ============================================================
# --- PROFILE BANK (dal DB) ---------------------------------------------------
def _add_tip(tips: list[str], msg: str):
    if msg and msg not in tips:
        tips.append(msg)

def _fmt_dbfs(v: float) -> str:
    # es. -18.23 -> "-18.2 dBFS"
    try:
        return f"{float(v):.1f} dBFS"
    except Exception:
        return str(v)

def _advices_from_metrics(
    *,
    is_tts: bool,
    speech_sec: float,
    speech_ratio: float,
    pause_count: int,
    snr_db: float,
    clip_ratio: float,
    rms_dbfs: float,
    hum50_db: float,
    hum60_db: float,
    dc_offset: float,
    spk_count: int,
    spk_conf: float,
    intra_mean: float,
    intra_p05: float,
    prof_sim: float | None,
    bank_size: int | None
) -> list[str]:
    tips: list[str] = []
    target_rms_lo, target_rms_hi = -18.0, -14.0
    hum_gate = 3.0  # dB di prominenza oltre il baseline
    multi_confident = (spk_count >= 2 and spk_conf >= 0.50)

    # --- Consapevolezza TTS vs registrazione reale ---
    if is_tts:
        # Durata utile (evita matching fragile)
        if speech_sec < 6.0:
            _add_tip(tips, f"Testo troppo corto per il match robusto: genera almeno 8–12 s di parlato continuo (ora {speech_sec:.1f}s).")

        # Densità parlato: evita silenzi lunghi in output TTS
        if speech_ratio < 0.7 or pause_count >= 3:
            _add_tip(tips, "Riduci i silenzi in TTS: usa `XTTS_MAX_SIL_MS≈250–400` e valuta `split_sentences=false` per frasi brevi.")

        # Coesione timbrica tra chunk
        if intra_p05 < 0.6:
            _add_tip(tips, "Il timbro oscilla tra i chunk: riusa i latents del profilo (`speaker_embedding`) invece di ricalcolarli ad ogni chiamata.")

        # Similarità al profilo
        if prof_sim is not None and prof_sim < 0.78:
            _add_tip(tips, "Il TTS non combacia bene col profilo: aggiungi più riferimenti `voice_ref` (5–15 min complessivi) e riallinea tono/velocità/lingua ai ref, poi re-enroll.")

        # Bank size (di solito 1: lo speaker attuale)
        if (bank_size or 0) < 1:
            _add_tip(tips, "Nessun profilo trovato: effettua l’enroll dei riferimenti voce prima del clone.")

        # Multi-speaker in output TTS => refs sporchi
        if multi_confident:
            _add_tip(tips, "Rilevata più di una voce: filtra i file marcati `voice_ref` (usa solo la tua voce) e ripeti l’enroll.")

        # Livelli e clipping (pipeline di post-processing)
        if clip_ratio > 0.0:
            _add_tip(tips, f"Clipping nel TTS: riduci il limiter/post-gain (clip≈{clip_ratio*100:.2f}%). Mantieni il picco a −1 dBFS.")
        if not (target_rms_lo <= rms_dbfs <= target_rms_hi):
            _add_tip(tips, f"Normalizza l’output TTS a RMS {target_rms_lo:.0f}…{target_rms_hi:.0f} dBFS (ora {_fmt_dbfs(rms_dbfs)}).")

        # Hum/DC (rarissimi in TTS, ma se ci sono è la catena audio)
        worst_hum = max(hum50_db, hum60_db)
        if worst_hum > hum_gate:
            which = "50 Hz" if hum50_db >= hum60_db else "60 Hz"
            _add_tip(tips, f"Ronzii di rete in catena audio ({which}): applica un notch {which} o verifica plugin/equalizzatori a monte del salvataggio.")
        if abs(dc_offset) > 0.02:
            _add_tip(tips, "DC offset in uscita TTS: abilita un high-pass a 20 Hz o rimuovi l’offset in post.")

    else:
        # --- Registrazione reale (microfono/stanza) ---
        if speech_sec < 8.0:
            _add_tip(tips, f"Registra almeno 8–12 secondi di parlato continuo (ora {speech_sec:.1f}s).")
        if snr_db < 20.0:
            _add_tip(tips, f"Ambiente rumoroso (SNR {snr_db:.1f} dB): avvicina il microfono, riduci rumori di fondo o cambia stanza.")
        if speech_ratio < 0.7:
            _add_tip(tips, "Taglia silenzi lunghi all’inizio/fine o riduci le pause.")
        if clip_ratio > 0.002:
            _add_tip(tips, f"Volume troppo alto: stai saturando (clip≈{clip_ratio*100:.2f}%). Abbassa il gain o allontana il microfono.")
        if rms_dbfs < -28.0:
            _add_tip(tips, f"Segnale debole ({_fmt_dbfs(rms_dbfs)}): avvicina il microfono o alza leggermente il gain.")
        if rms_dbfs > -10.0:
            _add_tip(tips, f"Segnale molto alto ({_fmt_dbfs(rms_dbfs)}): riduci il gain per evitare distorsione.")
        worst_hum = max(hum50_db, hum60_db)
        if worst_hum > hum_gate:
            which = "50 Hz" if hum50_db >= hum60_db else "60 Hz"
            _add_tip(tips, f"Possibile hum di rete {which}: usa un notch a {which} o verifica l’alimentazione/loop di massa.")
        if abs(dc_offset) > 0.02:
            _add_tip(tips, "DC offset elevato: applica un high-pass a 20 Hz o un filtro DC offset remover.")
        if (prof_sim is not None) and (prof_sim < 0.70):
            _add_tip(tips, "Il timbro differisce dal profilo: usa la stessa voce/registro/velocità dei riferimenti.")
        if multi_confident and intra_p05 <= 0.55:
            _add_tip(tips, "Più voci rilevate: registra da solo, in ambiente controllato.")

    # Dedup e ordine stabile
    return tips

def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v) + 1e-9)
    return v / n

def build_profile_bank(profiles: Dict[str, List[str]]) -> list[tuple[str, np.ndarray]]:
    """
    profiles: {profile_id: [path_ref1.wav, path_ref2.wav, ...]}
    Ritorna una lista di (profile_id, centroido L2-normalizzato).
    """
    bank: list[tuple[str, np.ndarray]] = []
    for pid, paths in profiles.items():
        c = speaker_centroid_from_paths(paths)
        if np.any(c):  # scarta centroidi nulli
            bank.append((pid, _normalize_vec(c)))
    return bank

def best_match_to_bank(embs: np.ndarray, bank: list[tuple[str, np.ndarray]]) -> tuple[str | None, float]:
    """
    Confronta il centroido della clip con i centroidi in bank.
    Ritorna (best_id, best_cos_sim). Se bank vuoto -> (None, -1.0).
    """
    if embs.ndim != 2 or embs.shape[0] == 0 or not bank:
        return None, -1.0
    c = _normalize_vec(np.mean(embs, axis=0))
    best_id, best_sim = None, -1.0
    for pid, cc in bank:
        sim = float(np.dot(c, cc))  # cos perché L2-normalizzati
        if sim > best_sim:
            best_sim, best_id = sim, pid
    return best_id, best_sim

def speech_sliding_chunks(
    wav: np.ndarray, sr: int, segs: list[tuple[int,int]],
    win_s: float = 1.5, hop_s: float = 0.75
) -> list[np.ndarray]:
    if not segs:
        return []
    total_speech = sum((e - s) for s, e in segs) / sr
    # per clip corti: torna segmenti "pieni" senza overlap (stop ai falsi cluster)
    if total_speech < 6.0:
        return cut_chunks(wav, sr, segs, max_chunk_sec=min(3.0, total_speech if total_speech>0 else 3.0))

    win = int(sr * win_s)
    hop = int(sr * hop_s)
    out: list[np.ndarray] = []
    for s, e in segs:
        a, b = int(s), int(e)
        if b - a < int(sr * 0.8):
            continue
        i = a
        while i + win <= b:
            out.append(wav[i:i+win])
            i += hop
        if b - i >= int(win * 0.6) and b - win >= a:
            out.append(wav[b - win:b])
    return out


def robust_speaker_count(embs: np.ndarray) -> tuple[int, float, np.ndarray | None, dict]:
    diag = {}
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim != 2 or embs.shape[0] == 0:
        return 0, 0.0, None, {"reason":"no_embeddings"}
    n = embs.shape[0]
    if n == 1:
        return 1, 0.9, np.zeros((1,), dtype=int), {"reason":"single_embedding"}

    # L2 normalize
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    mean_sim, p05_sim = _pairwise_cosine(embs)
    diag["pairwise_mean"] = float(mean_sim)
    diag["pairwise_p05"] = float(p05_sim)

    # Early-exit 1 speaker più permissivo per pochi chunk
    if n <= 4 and (mean_sim >= 0.70 or p05_sim >= 0.55):
        return 1, 0.8, np.zeros((n,), dtype=int), {**diag, "reason":"small_n_conservative"}

    # cluster su k=2..3
    best_k, best_score, best_labels = 1, -1.0, None
    tried = {}
    for k in (2, 3):
        if n < k: 
            continue
        cl = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = cl.fit_predict(embs)
        try:
            sil = silhouette_score(embs, labels, metric="cosine")
        except Exception:
            sil = -1.0
        tried[k] = float(sil)
        if sil > best_score:
            best_score, best_k, best_labels = sil, k, labels
    diag["silhouette"] = tried

    # gate silhouette dinamico: più alto se n è basso
    if n <= 4:
        sil_gate = 0.65
    elif n <= 8:
        sil_gate = 0.35
    else:
        sil_gate = 0.20

    if best_score < sil_gate:
        return 1, 1.0 - max(0.0, best_score), np.zeros((n,), dtype=int), {
            **diag, "reason":"low_silhouette", "silhouette_best":float(best_score), "sil_gate":sil_gate
        }

    # sbilanciamento cluster
    counts = np.bincount(best_labels)
    diag["counts"] = counts.tolist()
    imb = counts.max() / counts.sum()
    diag["imbalance"] = float(imb)
    if imb >= 0.90 or counts.min() < 2:
        return 1, 0.8, np.zeros((n,), dtype=int), {**diag, "reason":"imbalanced_clusters"}

    return best_k, float(min(1.0, max(0.2, best_score))), best_labels, {**diag, "reason":"multi"}

def _dbfs_metrics(wav: np.ndarray) -> dict:
    """
    Livelli base in dBFS e qualità del segnale grezza.
    """
    if wav.size == 0:
        return {"rms_dbfs": -np.inf, "peak_dbfs": -np.inf, "crest_db": 0.0, "dc_offset": 0.0, "zcr": 0.0}
    peak = float(np.max(np.abs(wav)) + 1e-12)
    rms = float(np.sqrt(np.mean(wav**2)) + 1e-12)
    rms_dbfs = 20.0 * np.log10(rms)
    peak_dbfs = 20.0 * np.log10(peak)
    crest = peak_dbfs - rms_dbfs
    dc = float(np.mean(wav))
    # ZCR (zero-crossing rate) grezzo: correlato a fruscio/rumore
    zcr = float(np.mean(np.abs(np.diff(np.signbit(wav)).astype(np.float32))))
    return {
        "rms_dbfs": float(rms_dbfs),
        "peak_dbfs": float(peak_dbfs),
        "crest_db": float(crest),
        "dc_offset": float(dc),
        "zcr": float(zcr),
    }

def _hum_buzz_scores(wav: np.ndarray, sr: int) -> dict:
    """
    Stima rozza della presenza di hum 50/60 Hz (+ armoniche fino a ~400 Hz).
    Ritorna 'hum50_db', 'hum60_db' come prominenza (dB) rispetto alla banda 20–1000 Hz.
    """
    if wav.size < sr//2:
        return {"hum50_db": 0.0, "hum60_db": 0.0}

    # --- FIX: window sulla lunghezza reale L e FFT zero-padded a N ---
    wav = np.asarray(wav, dtype=np.float32)
    L = wav.shape[0]

    # prossima potenza di 2 >= L (per FFT efficiente)
    N = 1
    while N < L:
        N <<= 1

    # finestra Hann su L e zero-padding a N
    windowed = wav * np.hanning(L).astype(np.float32)
    spec = np.abs(np.fft.rfft(windowed, n=N)) + 1e-9
    freqs = np.fft.rfftfreq(N, 1.0 / sr)

    def band_db(center, width=2.0):
        m = (freqs >= max(20.0, center - width)) & (freqs <= center + width)
        return float(10 * np.log10(np.sum(spec[m]) + 1e-12))

    def baseline_db(lo=20.0, hi=1000.0, notches=None):
        m = (freqs >= lo) & (freqs <= hi)
        s = np.copy(spec[m])
        if notches:
            nf = freqs[m]
            for c, w in notches:
                mask = (nf >= c - w) & (nf <= c + w)
                s[mask] = 0.0
        return float(10 * np.log10(np.sum(s) + 1e-12))

    def hum_prominence(base, f0):
        # somma 1°..6° armonica strette
        bands = [(f0 * i, 2.0 + 0.2 * i) for i in range(1, 7) if f0 * i <= 400.0]
        s_h = sum(10 ** (band_db(c, w) / 10.0) for c, w in bands)
        hum_db = 10 * np.log10(s_h + 1e-12)
        return hum_db - base

    base = baseline_db(notches=[(50, 3), (60, 3), (100, 3), (120, 3), (150, 3), (180, 3), (200, 3), (240, 3)])
    prom50 = hum_prominence(base, 50.0)
    prom60 = hum_prominence(base, 60.0)
    return {"hum50_db": float(max(0.0, prom50)), "hum60_db": float(max(0.0, prom60))}

# ============================================================================

def grade_clip(
    path: str,
    profile_centroid: np.ndarray | None = None,
    profile_bank: list[tuple[str, np.ndarray]] | None = None,  # <-- NEW
    is_tts: bool = False                                       # <-- NEW
) -> dict[str, any]:
    """
    Valuta la clip con molte più feature + stima parlanti robusta.
    Restituisce:
      - score/label + breakdown con pesi/penalità
      - metriche diagnostiche (livelli in dBFS, hum, ecc.)
      - confidenza multi-speaker e dettagli clustering (diagnostics.speakers)
    """
    # --- load + feature base ---
    wav, sr = load_audio_mono_16k(path)
    dur = float(len(wav)) / sr if sr else 0.0

    env = _envelope(wav, sr, ms=20.0)
    snr_db = _snr_db_from_env(env)
    clip_ratio = _clip_ratio(wav)
    levels = _dbfs_metrics(wav)
    hum = _hum_buzz_scores(wav, sr)

    # --- VAD ---
    segs = vad_segments(
        wav, sr,
        frame_ms=30, aggressiveness=2,
        min_speech_ms=350, max_gap_ms=250
    )
    speech_samples = sum(max(0, e - s) for s, e in segs)
    speech_sec = speech_samples / sr if sr else 0.0
    speech_ratio = (speech_sec / dur) if dur > 0 else 0.0
    pause_count = max(0, len(segs) - 1)

    # --- Chunks + embeddings (sliding windows uniformi per robustezza) ---
    chunks = speech_sliding_chunks(wav, sr, segs, win_s=1.5, hop_s=0.75)
    if not chunks:
        # fallback: usa il vecchio slicer se VAD è troppo avaro
        chunks = cut_chunks(wav, sr, segs, max_chunk_sec=5.0)

    embs = embed_chunks(chunks, sr=sr)
    embs = np.asarray(embs)
    if embs.ndim != 2:
        embs = embs.reshape(embs.shape[0], -1)

    intra_mean, intra_p05 = _pairwise_cosine(embs)

    # --- Multi-speaker robusto ---
    intra_mean, intra_p05 = _pairwise_cosine(embs)

    # --- Multi-speaker robusto ---
    if embs.shape[0] >= 2:
        spk_count, spk_conf, spk_labels, spk_diag = robust_speaker_count(embs)
    else:
        spk_count = 1 if speech_sec > 0.5 else 0
        spk_conf, spk_labels, spk_diag = 0.5, None, {"reason":"not_enough_chunks"}

    # small-sample gate
    best_sil = float(max(spk_diag.get("silhouette", {}).values() or [-1.0]))
    small_sample = (len(chunks) < 6) or (len(segs) <= 1) or (speech_sec < 6.0)
    if spk_count >= 2 and small_sample and best_sil < 0.75:
        spk_diag = {**spk_diag, "reason": "overruled_small_sample", "silhouette_best": best_sil}
        spk_count, spk_conf, spk_labels = 1, max(0.3, 1.0 - spk_conf), None

    # --- Similarità al profilo (singolo centroido oppure bank) ------------
    prof_sim = None
    matched_profile_id = None

    if profile_bank:
        matched_profile_id, best_sim = best_match_to_bank(embs, profile_bank)
        if best_sim >= 0:  # setta prof_sim dal bank
            prof_sim = float(best_sim)

        # OVERRIDE per TTS: se il TTS matcha forte il profilo utente, forza 1 speaker
        # soglia prudente: 0.78 (puoi tararla 0.75..0.82 in base ai tuoi dati)
        if is_tts and best_sim >= 0.78:
            spk_diag = {**spk_diag, "reason": "tts_profile_match", "matched_profile": matched_profile_id, "best_sim": best_sim}
            spk_count, spk_conf, spk_labels = 1, max(spk_conf, 0.9), None

    # fallback: vecchio parametro singolo
    if prof_sim is None and profile_centroid is not None and profile_centroid.shape == (192,) and embs.shape[0] > 0:
        c = np.mean(embs, axis=0); c /= (np.linalg.norm(c) + 1e-9)
        prof_sim = float(np.dot(c, profile_centroid))

    # --- Sub-score (punteggi parziali) ------------------------------------
    # Pesi: 25% durata utile, 20% SNR, 20% stabilità timbro, 20% match profilo, 15% livello
    len_score  = min(1.0, speech_sec / 10.0) * 100.0
    snr_score  = _map_piecewise(snr_db, [5, 15, 25, 35], [10, 50, 85, 100])
    stab_score = _map_piecewise(intra_mean, [0.40, 0.60, 0.75, 0.85], [20, 60, 85, 100])
    prof_score = _map_piecewise(prof_sim if prof_sim is not None else 0.70,
                                [0.55, 0.70, 0.80, 0.90], [10, 50, 85, 100])
    # livello ideale: RMS ~ -18..-14 dBFS
    rms = levels["rms_dbfs"]
    level_score = _map_piecewise(rms, [-40, -28, -22, -18, -14, -10], [20, 55, 80, 100, 90, 65])

    # --- Penalità (applicate con buon senso) ------------------------------
    # multi-speaker: penalizza solo se confidente e poca coesione (intra_p05 basso)
    multi_confident = (spk_count >= 2 and spk_conf >= 0.50 and not small_sample)
    pen_multi = 0.0
    if multi_confident and intra_p05 <= 0.55:  # un filo più severo
        pen_multi = 20.0 + 10.0 * (spk_count - 1)

    # parlato poco denso nel file
    pen_ratio = 0.0 if speech_ratio >= 0.7 else 20.0 * ((0.7 - speech_ratio) / 0.7)

    # clipping
    pen_clip = min(30.0, 200.0 * clip_ratio)

    # hum (prendi il peggiore tra 50/60 Hz) – fino a 15 punti
    hum_prom = max(hum["hum50_db"], hum["hum60_db"])
    pen_hum = float(min(15.0, max(0.0, (hum_prom - 3.0) * 3.0)))  # gate ~3 dB sopra baseline

    # DC offset – fino a 10 pt
    pen_dc = float(min(10.0, (abs(levels["dc_offset"]) / 0.03) * 10.0))

    # --- Score finale ------------------------------------------------------
    score = (
        0.25 * len_score +
        0.20 * snr_score +
        0.20 * stab_score +
        0.20 * prof_score +
        0.15 * level_score -
        pen_multi - pen_ratio - pen_clip - pen_hum - pen_dc
    )
    score = float(max(0.0, min(100.0, score)))
    label = _label_from_score(score)

        # --- Suggerimenti pratici (consapevoli di TTS vs reale) ---------------
    bank_sz = len(profile_bank) if profile_bank else 0
    tips = _advices_from_metrics(
        is_tts=is_tts,
        speech_sec=speech_sec,
        speech_ratio=speech_ratio,
        pause_count=pause_count,
        snr_db=snr_db,
        clip_ratio=clip_ratio,
        rms_dbfs=levels["rms_dbfs"],
        hum50_db=hum["hum50_db"],
        hum60_db=hum["hum60_db"],
        dc_offset=levels["dc_offset"],
        spk_count=spk_count,
        spk_conf=spk_conf,
        intra_mean=intra_mean,
        intra_p05=intra_p05,
        prof_sim=prof_sim,
        bank_size=bank_sz,
    )


    return {
        "ok": True,
        # — riepilogo —
        "duration_sec": round(dur, 2),
        "speech_sec": round(speech_sec, 2),
        "speech_ratio": round(speech_ratio, 3),
        "snr_db": round(float(snr_db), 2),
        "clip_ratio": round(float(clip_ratio), 4),

        # — parlanti —
        "speakers": int(spk_count),
        "speakers_confidence": round(float(spk_conf), 3),
        "intra_cosine_mean": round(float(intra_mean), 3),
        "intra_cosine_p05": round(float(intra_p05), 3),
        "profile_sim": (round(float(prof_sim), 3) if prof_sim is not None else None),

        # — livelli/rumori —
        "rms_dbfs": round(levels["rms_dbfs"], 2),
        "peak_dbfs": round(levels["peak_dbfs"], 2),
        "crest_db": round(levels["crest_db"], 2),
        "dc_offset": round(levels["dc_offset"], 4),
        "hum50_db": round(hum["hum50_db"], 2),
        "hum60_db": round(hum["hum60_db"], 2),
        "pause_count": int(pause_count),
        "chunks": int(len(chunks)),

        # — valutazione —
        "score": round(score, 1),
        "label": label,

        # — consigli —
        "advices": tips,

        # — scomposizione con pesi/penalità —
        "breakdown": {
            "weights": {"len":0.25, "snr":0.20, "stability":0.20, "profile":0.20, "level":0.15},
            "len": round(len_score, 1),
            "snr": round(snr_score, 1),
            "stability": round(stab_score, 1),
            "profile": round(prof_score, 1),
            "level": round(level_score, 1),
            "penalties": {
                "multi": round(pen_multi, 1),
                "speech_ratio": round(pen_ratio, 1),
                "clipping": round(pen_clip, 1),
                "hum": round(pen_hum, 1),
                "dc_offset": round(pen_dc, 1),
            }
        },
        # — diagnostica utile a debug/telemetria —
        "diagnostics": {
            "vad": {"segments": len(segs), "params": {"frame_ms":30,"aggr":2,"min_ms":350,"max_gap_ms":250}},
            "speakers": spk_diag,
             "profile": {
                "matched_id": matched_profile_id,
                "bank_size": (len(profile_bank) if profile_bank else 0)
            },
            "thresholds": {
                "multi_penalty_conf_min": 0.25,
                "multi_penalty_p05_max": 0.60,
                "speech_ratio_target": 0.70,
                "hum_gate_db": 3.0,
                "dc_offset_ref": 0.03,
                "tts_profile_gate": 0.78
            }
        }
    }
