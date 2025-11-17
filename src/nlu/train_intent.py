# src/nlu/train_intent.py
import os
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np

DATA_CSV = "experiments/data/intent_data.csv"
OUT_DIR = "experiments/models/intent"
os.makedirs(OUT_DIR, exist_ok=True)

def load_dataset():
    df = pd.read_csv(DATA_CSV)
    # ensure label mapping
    labels = sorted(df["intent"].unique())
    label2id = {l:i for i,l in enumerate(labels)}
    df["labels"] = df["intent"].map(label2id)
    ds = Dataset.from_pandas(df[["text","labels"]])
    return ds, labels, label2id

def preprocess(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

def main():
    ds, labels, label2id = load_dataset()
    num_labels = len(labels)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized = ds.map(lambda x: preprocess(tokenizer, x), batched=True)
    tokenized = tokenized.train_test_split(test_size=0.1, seed=42)
    tokenized.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

    args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    eval_strategy="epoch",      # ✅ updated
    save_strategy="epoch",      # ✅ keep same
    logging_steps=50,
    seed=42,
    fp16=False                  # leave False for CPU
)


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    # save label mapping
    import json
    with open(os.path.join(OUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print("Saved intent model to", OUT_DIR)

if __name__ == "__main__":
    main()
