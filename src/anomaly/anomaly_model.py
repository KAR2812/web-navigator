# src/anomaly/anomaly_model.py
import os
from joblib import load
import numpy as np
import pandas as pd

MODEL = "experiments/models/anomaly.joblib"
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL):
        return None
    _model = load(MODEL)
    return _model

def _featurize_one(trace):
    # same features as training
    x = {}
    x["num_elements"] = float(trace.get("num_elements",0))
    x["avg_text_len"] = float(trace.get("avg_text_len",0.0))
    x["step_idx"] = float(trace.get("step_idx",0))
    x["step_duration"] = float(trace.get("step_duration",0.0))
    x["action_len"] = float(len(str(trace.get("action_type",""))))
    x["url_hash"] = float(hash(str(trace.get("url",""))) % 1000)
    return pd.DataFrame([x])

def is_likely_fail(trace: dict, threshold: float = 0.5):
    model = load_model()
    if model is None:
        return False, 0.0
    X = _featurize_one(trace)
    try:
        probs = model.predict_proba(X)[:,1]
        p = float(probs[0])
    except Exception:
        p = float(model.predict(X)[0])
    return p < threshold, p
