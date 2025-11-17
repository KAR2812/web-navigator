# src/speech/voice_to_text.py
"""
Record microphone audio (until Enter pressed), save to a temporary WAV,
and transcribe using OpenAI Whisper (python package 'whisper').

Usage:
    from src.speech.voice_to_text import listen_and_transcribe
    text = listen_and_transcribe(model_name="base", sample_rate=16000)
"""
from src.speech.translate import translate_to_en, detect_language
import tempfile
import sys
import time
import sounddevice as sd
import soundfile as sf
import whisper
import os
from typing import Optional

def record_to_wav(path: str, samplerate: int = 16000, channels: int = 1):
    """
    Record from default microphone until the user presses Enter.
    Saves to `path` as 16-bit WAV.
    """
    print("Recording... Press Enter to stop recording.")
    # start a non-blocking stream and append in callback
    data = []

    def callback(indata, frames, time_info, status):
        if status:
            print("Recording status:", status, file=sys.stderr)
        data.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        try:
            # wait for user to press Enter
            input()
        except KeyboardInterrupt:
            # allow Ctrl+C to stop
            pass

    # concatenate and write file
    import numpy as np
    recording = np.concatenate(data, axis=0)
    sf.write(path, recording, samplerate)
    print(f"Saved recording to: {path}")

def listen_and_transcribe(model_name: str = "base", samplerate: int = 16000, verbose: bool = False,
                         force_language: Optional[str] = None, translate_after: bool = True) -> str:
    """
    Record audio then transcribe using whisper.
    - force_language: if provided (e.g., "en"), whisper transcribes in that language.
    - translate_after: if True, detect hi/mr in transcript and translate to English.
    Returns final text (English if translated).
    """
    # create temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name

    try:
        # record
        record_to_wav(wav_path, samplerate=samplerate, channels=1)

        # load whisper model (cached)
        if verbose:
            print(f"Loading Whisper model '{model_name}' (this may take a moment)...")
        model = whisper.load_model(model_name)

        if verbose:
            print("Transcribing...")

        options = {}
        if force_language:
            options["language"] = force_language  # e.g., "en"
            # do not set task=translate here; we'll use external translator if needed

        result = model.transcribe(wav_path, **options)
        text = result.get("text", "").strip()
        if verbose:
            print("Raw transcription:", repr(text))

        if translate_after:
            # detect and translate only hi/mr
            lang = detect_language(text)
            if lang:
                lang = lang.lower()
            if lang and (lang.startswith("hi") or lang.startswith("mr")):
                if verbose:
                    print("Detected language:", lang, " — translating to English...")
                translated = translate_to_en(text, src_lang=lang[:2])
                if verbose:
                    print("Translation result:", repr(translated))
                return translated

        return text

    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass