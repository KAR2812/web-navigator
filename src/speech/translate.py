# src/speech/translate.py
"""
Translation helpers for Marathi/Hindi -> English.

Provides:
  - detect_language(text) -> 'hi' | 'mr' | other lang code or None
  - translate_to_en(text, src_lang=None) -> English translation (or original text if unsupported)

Implements efficient cached model loading and safe chunking using Hugging Face transformers.
"""

from typing import Optional, List
import re
import threading
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# map short langs to Marian models
_MODEL_MAP = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "mr": "Helsinki-NLP/opus-mt-mr-en",
}

_CACHE = {}
_LOCK = threading.Lock()

_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def detect_language(text: str) -> Optional[str]:
    """Return short language code like 'hi', 'mr', 'en', etc., or None on failure."""
    if not text or not text.strip():
        return None
    try:
        code = detect(text)
        return code
    except LangDetectException:
        return None
    except Exception:
        return None

def _get_tokenizer_and_model(lang: str):
    """Load and cache tokenizer+model for 'hi' or 'mr'."""
    lang = (lang or "")[:2].lower()
    if lang not in _MODEL_MAP:
        raise ValueError(f"No model for language '{lang}'")
    model_name = _MODEL_MAP[lang]
    with _LOCK:
        if model_name in _CACHE:
            return _CACHE[model_name]["tokenizer"], _CACHE[model_name]["model"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.config.max_length = 512
        _CACHE[model_name] = {"tokenizer": tokenizer, "model": model}
        return tokenizer, model

def _chunk_text_for_tokenizer(text: str, tokenizer, max_tokens: int = 400) -> List[str]:
    """Chunk by sentence boundaries and token budget to avoid overflow."""
    if not text:
        return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]
    parts = _SPLIT_RE.split(text)
    chunks = []
    cur = ""
    cur_len = 0
    for part in parts:
        if not part.strip():
            continue
        part_tokens = tokenizer.encode(part, add_special_tokens=False)
        pl = len(part_tokens)
        if cur_len + pl <= max_tokens:
            cur = (cur + " " + part).strip() if cur else part
            cur_len += pl
        else:
            if cur:
                chunks.append(cur.strip())
            if pl > max_tokens:
                toks = tokenizer.encode(part, add_special_tokens=False)
                i = 0
                while i < len(toks):
                    slice_toks = toks[i:i+max_tokens]
                    piece = tokenizer.decode(slice_toks, clean_up_tokenization_spaces=True)
                    chunks.append(piece.strip())
                    i += max_tokens
                cur = ""
                cur_len = 0
            else:
                cur = part
                cur_len = pl
    if cur:
        chunks.append(cur.strip())
    return chunks

def translate_to_en(text: str, src_lang: Optional[str] = None, num_beams: int = 3, max_length: int = 256) -> str:
    """
    Translate text to English.
    - src_lang: optional short code like 'hi' or 'mr'. If None, caller should detect.
    - Returns English translation or original text when unsupported.
    """
    if not text or not text.strip():
        return text or ""
    lang = (src_lang or "").lower()[:2] if src_lang else None
    if lang is None:
        # try to detect via langdetect
        detected = detect_language(text)
        if detected:
            lang = detected[:2].lower()
    if lang not in _MODEL_MAP:
        # unsupported language — return original text
        return text

    tokenizer, model = _get_tokenizer_and_model(lang)
    chunks = _chunk_text_for_tokenizer(text, tokenizer, max_tokens=400)
    out_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        gen = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        decoded = tokenizer.decode(gen[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        out_chunks.append(decoded.strip())
    return " ".join([c for c in out_chunks if c])

# Backwards compatibility: alias names some previous code expected
translate_text = translate_to_en
translate_auto = translate_to_en
