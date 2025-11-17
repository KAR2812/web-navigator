# src/anomaly/logger.py
import csv
import os
import time

OUT = "experiments/data/traces.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

HEADER = ["timestamp","run_id","step_idx","url","num_elements","avg_text_len","action_type","step_duration","reward","success_after_n"]

def log_trace(trace: dict):
    """
    trace keys:
    run_id, step_idx (int), url, num_elements (int), avg_text_len (float),
    action_type (str), step_duration (float), reward (float), success_after_n (0/1)
    """
    write_header = not os.path.exists(OUT)
    row = [
        int(time.time()),
        trace.get("run_id",""),
        int(trace.get("step_idx",0)),
        trace.get("url","")[:200],
        int(trace.get("num_elements",0)),
        float(trace.get("avg_text_len",0.0)),
        trace.get("action_type","")[:50],
        float(trace.get("step_duration",0.0)),
        float(trace.get("reward",0.0)),
        int(trace.get("success_after_n",0))
    ]
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(HEADER)
        writer.writerow(row)
