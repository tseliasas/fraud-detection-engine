import json
import pandas as pd
from data_prep import load_data, clean, split

# Real fraud rate is ~0.17% (1 in 578), far too rare to watch live,
# so the demo keeps every test fraud and samples enough legit rows for ~5% fraud
LEGIT_PER_FRAUD = 20
SEED = 42
OUTPUT_PATH = "../dashboard/public/demo_transactions.json"

def build_demo_set(X_test, y_test):
    frauds = X_test[y_test == 1]
    legit  = X_test[y_test == 0].sample(len(frauds) * LEGIT_PER_FRAUD, random_state=SEED)

    demo = pd.concat([frauds.assign(actualFraud=True), legit.assign(actualFraud=False)])
    return demo.sample(frac=1, random_state=SEED)

def to_records(demo):
    # The API computes hour from the live timestamp, so it isn't sent as a feature
    feature_columns = [c for c in demo.columns if c not in ("hour", "actualFraud")]

    records = []
    for i, (_, row) in enumerate(demo.iterrows()):
        records.append({
            "id": i,
            "actualFraud": bool(row["actualFraud"]),
            "features": {c: round(float(row[c]), 6) for c in feature_columns},
        })
    return records

if __name__ == "__main__":
    df = load_data("../data/raw/creditcard.csv")
    df = clean(df)

    # Only test rows: transactions the model never saw during training or threshold tuning
    X_train, X_test, y_train, y_test = split(df, 0.2, 42)

    records = to_records(build_demo_set(X_test, y_test))

    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, separators=(",", ":"))  # no spaces: smaller download for the browser

    fraud_count = sum(r["actualFraud"] for r in records)
    print(f"Saved {len(records)} transactions ({fraud_count} fraud) to {OUTPUT_PATH}")
