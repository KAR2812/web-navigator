# src/speech/online_transcribe.py
import speech_recognition as sr

def online_transcribe(timeout=5, phrase_time_limit=10):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        text = r.recognize_google(audio)
        print("Google STT:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print("API error:", e)
        return ""
