# /root/ml-serve/realtime_ws.py
import os
import asyncio
import json

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from faster_whisper import WhisperModel

MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")

HOST = os.getenv("ASR_WS_HOST", "0.0.0.0")
PORT = int(os.getenv("ASR_WS_PORT", "8765"))

SAMPLE_RATE = 16000

print(f"[WS-ASR] loading Whisper model: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


async def asr_handler(ws):
    """
    CLIENT → SERVER:
      - frame binari: PCM int16 LE mono 16k
      - stringa 'end' quando ha finito di parlare

    SERVER → CLIENT:
      - {"type": "partial", "text": "..."}
      - {"type": "final",   "text": "..."}
    """
    print(f"[WS-ASR] client connected: {ws.remote_address}")
    audio_buffer = np.zeros(0, dtype=np.float32)

    try:
        async for msg in ws:
            # --- AUDIO BINARIO ---
            if isinstance(msg, (bytes, bytearray)):
                pcm_int16 = np.frombuffer(msg, dtype=np.int16)
                if pcm_int16.size == 0:
                    continue
                pcm = pcm_int16.astype(np.float32) / 32768.0
                audio_buffer = np.concatenate((audio_buffer, pcm))

                # parziale veloce ogni ~0.5s di audio
                if audio_buffer.shape[0] > SAMPLE_RATE * 0.5:
                    segments, _ = model.transcribe(
                        audio_buffer,
                        language="it",
                        beam_size=1,
                        vad_filter=True,
                        condition_on_previous_text=False,
                    )
                    partial_text = "".join(s.text for s in segments).strip()
                    if partial_text:
                        await ws.send(json.dumps({
                            "type": "partial",
                            "text": partial_text,
                        }))

            # --- MESSAGGIO TESTUALE ("end") ---
            elif isinstance(msg, str):
                m = msg.strip().lower()
                if m == "end":
                    dur = audio_buffer.shape[0] / SAMPLE_RATE
                    print(f"[WS-ASR] ricevuto 'end' (dur ≈ {dur:.2f}s)")
                    break
                else:
                    print(f"[WS-ASR] msg testo sconosciuto: {msg!r}")

        # === FINALE ===
        final_text = ""
        if audio_buffer.shape[0] > 0:
            segments, _ = model.transcribe(
                audio_buffer,
                language="it",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=True,
            )
            final_text = "".join(s.text for s in segments).strip()

        await ws.send(json.dumps({
            "type": "final",
            "text": final_text,
        }))

    except (ConnectionClosedOK, ConnectionClosedError) as e:
        # chiusura brusca lato client → non è “grave”
        print(f"[WS-ASR] conn chiusa dal client: {e}")
    except Exception as e:
        print("[WS-ASR] error:", e)
        try:
            await ws.send(json.dumps({
                "type": "error",
                "message": str(e),
            }))
        except Exception:
            pass
    finally:
        print(f"[WS-ASR] client disconnected: {ws.remote_address}")
        try:
            await ws.close()
        except Exception:
            pass


async def main():
    print(f"[WS-ASR] server listening on ws://{HOST}:{PORT}")
    async with websockets.serve(asr_handler, HOST, PORT, max_size=2**25):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
