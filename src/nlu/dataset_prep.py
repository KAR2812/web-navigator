# src/nlu/dataset_prep.py
import csv
import os
import random

OUT = "experiments/data/intent_data.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

INTENTS = {
    "search_product": [
        "find me {item}",
        "search for {item}",
        "show me {item} results",
        "i want to buy {item}"
    ],
    "login": [
        "log me in",
        "sign in to my account",
        "i want to login",
        "open my account"
    ],
    "book_ticket": [
        "book a ticket to {place} on {date}",
        "i want to book a ticket for {place}",
        "reserve a seat to {place}"
    ],
    "open_dashboard": [
        "open dashboard",
        "show me the admin dashboard",
        "go to dashboard"
    ],
    "logout": [
        "log out",
        "sign me out",
        "logout"
    ]
}

ITEMS = ["macbook", "iphone 14", "samsung tv", "headphones"]
PLACES = ["Mumbai", "Delhi", "Bengaluru", "Chennai"]
DATES = ["tomorrow", "next monday", "2025-08-01"]

SAMPLES_PER_INTENT = 150

rows = []
for intent, templates in INTENTS.items():
    for _ in range(SAMPLES_PER_INTENT):
        t = random.choice(templates)

        # ✅ Safe placeholder formatting
        fill = {
            "item": random.choice(ITEMS),
            "place": random.choice(PLACES),
            "date": random.choice(DATES)
        }
        for key in ["item", "place", "date"]:
            if "{" + key + "}" in t:
                t = t.format(**fill)

        if random.random() < 0.2:
            t = t + " please"

        rows.append((t, intent))


# Add some noisy/short examples
rows += [("help", "open_dashboard"), ("hello", "open_dashboard"), ("buy phone", "search_product")]

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "intent"])
    writer.writerows(rows)

print("Wrote", len(rows), "rows to", OUT)
