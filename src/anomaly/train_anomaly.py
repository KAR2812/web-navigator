# src/anomaly/train_anomaly.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump
import numpy as np

CSV = "experiments/data/traces.csv"
OUT = "experiments/models/anomaly.joblib"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def featurize(df):
    X = pd.DataFrame()
    X["num_elements"] = df["num_elements"].fillna(0).astype(float)
    X["avg_text_len"] = df["avg_text_len"].fillna(0).astype(float)
    X["step_idx"] = df["step_idx"].fillna(0).astype(float)
    X["step_duration"] = df["step_duration"].fillna(0).astype(float)
    # simple encoding of action_type: length + hash
    X["action_len"] = df["action_type"].fillna("").apply(len).astype(float)
    X["url_hash"] = df["url"].fillna("").apply(lambda s: hash(s) % 1000).astype(float)
    return X

def main():
    if not os.path.exists(CSV):
        print("No traces file at", CSV)
        return
    df = pd.read_csv(CSV)
    if "success_after_n" not in df.columns:
        print("No label column 'success_after_n' in traces. You must label data.")
        return
    X = featurize(df)
    y = df["success_after_n"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # try LightGBM first
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs)
        print("LightGBM AUC:", auc)
    except Exception as e:
        print("LightGBM not available or error:", e)
        print("Falling back to RandomForest")
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs)
        print("RandomForest AUC:", auc)

    print(classification_report(y_test, (probs>0.5).astype(int)))
    dump(model, OUT)
    print("Saved anomaly model to", OUT)

if __name__ == "__main__":
    main()
