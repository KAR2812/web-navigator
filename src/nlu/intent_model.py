# src/nlu/intent_model.py
import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "experiments/models/intent"

_tokenizer = None
_model = None
_labels = None

def _load():
    global _tokenizer, _model, _labels
    if _model is not None:
        return
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError("Intent model not found at " + MODEL_DIR)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    with open(os.path.join(MODEL_DIR, "labels.json"), "r", encoding="utf-8") as f:
        _labels = json.load(f)

def predict_intent(text: str):
    _load()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return _labels[idx], float(probs[idx])
