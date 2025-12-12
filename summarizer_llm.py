 # /root/ml-serve/summarizer_llm.py
import os, re, json, requests
from dataclasses import dataclass
from typing import List, Optional

# === Config via ENV (default sicuri) ===
LLM_BASE   = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
LLM_KEY    = os.getenv("LLM_API_KEY", "sk-void")
LLM_MODEL  = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
TEMP       = float(os.getenv("SUM_TEMP", "0.1"))
TOP_P      = float(os.getenv("SUM_TOP_P", "0.9"))
MAX_WORDS  = int(os.getenv("SUM_MAX_WORDS", "180"))
CHARS      = int(os.getenv("SUM_CHUNK_CHARS", "7000"))
OVERLAP    = int(os.getenv("SUM_CHUNK_OVERLAP", "900"))

_SYSTEM = (
  "Sei un assistente che riassume trascrizioni in ITALIANO con massima fedeltà.\n"
  "Regole: mantieni fatti, numeri, date e nomi; niente invenzioni; niente opinioni.\n"
  "Se ci sono to-do/decisioni/punti aperti, evidenziali. Stile chiaro e compatto."
)

@dataclass
class Summary:
    summary: str
    bullets: Optional[List[str]]
    model: str

def _strip_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s*\[?\(?\d{1,2}:\d{2}(?::\d{2})?\)?\]?\s*", " ", t)  # rimuove timecodes
    t = re.sub(r"\s{2,}", " ", t)
    return t

def _sentences(s: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+", s)
    return [p.strip() for p in parts if p.strip()]

def _chunk_by_chars(text: str, max_chars: int, overlap: int) -> List[str]:
    sents = _sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) + 1 > max_chars and cur:
            chunks.append(" ".join(cur))
            # soft-overlap: tieni ultime frasi fino a 'overlap'
            keep, tot = [], 0
            for ss in reversed(cur):
                if tot + len(ss) <= overlap:
                    keep.insert(0, ss); tot += len(ss)
                else:
                    break
            cur, cur_len = keep, sum(len(x) + 1 for x in keep)
        cur.append(s); cur_len += len(s) + 1
    if cur: chunks.append(" ".join(cur))
    return chunks

def _call_llm(prompt: str, out_json: bool) -> str:
    url = f"{LLM_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_KEY}"}
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMP,
        "top_p": TOP_P,
        "max_tokens": min(2048, MAX_WORDS * 4),
    }
    if out_json:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def _map_one(text: str, lang: str, target_words: int) -> dict:
    ask = (
      f"Testo da riassumere ({lang}):\n---\n{text}\n---\n"
      f"Scrivi un riassunto in {lang} di ~{target_words} parole, altissima precisione. "
      'Restituisci JSON con chiavi "summary" e opzionale "bullets" (max 8).'
    )
    raw = _call_llm(ask, out_json=True)
    try: obj = json.loads(raw)
    except: obj = {"summary": raw.strip()}
    if not isinstance(obj, dict): obj = {"summary": str(obj)}
    bl = obj.get("bullets")
    obj["bullets"] = [str(x).strip() for x in (bl or []) if str(x).strip()][:8] or None
    return obj

def summarize(text: str, lang: str = "it", style: str = "paragraph", max_words: int = MAX_WORDS) -> Summary:
    base = _strip_text(text)
    if not base:
        return Summary(summary="", bullets=None, model=f"vLLM:{LLM_MODEL}")

    # MAP
    parts = []
    for ch in _chunk_by_chars(base, CHARS, OVERLAP):
        parts.append(_map_one(ch, lang, max_words))

    glue = "\n\n".join([p["summary"] for p in parts if p.get("summary")])
    bullets_flat = []
    for p in parts:
        if p.get("bullets"):
            bullets_flat.extend(p["bullets"])

    # REDUCE
    reduce_prompt = (
      f"Unisci e compatta i seguenti riassunti parziali in {lang} (senza perdere fatti):\n"
      f"---\n{glue}\n---\n"
      f"Produci un riassunto unico (≈{max_words} parole). "
      'Restituisci JSON {"summary": string, "bullets": array opzionale max 8}.'
    )
    raw = _call_llm(reduce_prompt, out_json=True)
    try: final = json.loads(raw)
    except: final = {"summary": raw.strip(), "bullets": None}

    if (not final.get("bullets")) and bullets_flat:
        seen, uniq = set(), []
        for b in bullets_flat:
            k = b.lower()
            if k in seen: continue
            seen.add(k); uniq.append(b)
            if len(uniq) >= 8: break
        final["bullets"] = uniq

    text_out = str(final.get("summary", "")).strip()
    bl_out = final.get("bullets") or None
    if isinstance(bl_out, list):
        bl_out = [str(x).strip(" -•\t") for x in bl_out if str(x).strip()][:8]
    else:
        bl_out = None

    if style == "bullets" and bl_out:
        return Summary(summary="\n".join(f"• {b}" for b in bl_out), bullets=bl_out, model=f"vLLM:{LLM_MODEL}")
    return Summary(summary=text_out, bullets=bl_out, model=f"vLLM:{LLM_MODEL}") 