# Fraud Detection Engine

[![CI](https://github.com/tseliasas/fraud-detection-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/tseliasas/fraud-detection-engine/actions/workflows/ci.yml)

A credit card fraud detector built end to end: a model trained in Python, served by a C# API, and watched live on a Next.js dashboard.

You give it a transaction, and it tells you how likely it is to be fraud and whether it should be flagged. That takes about a millisecond.

![Dashboard](docs/dashboard.png)

## Results

Scored once on a held-out test set of 56,746 transactions the model never saw:

| Metric | Value |
|---|---|
| PR-AUC | 0.822 |
| Recall | 74.7% (caught 71 of 95 frauds) |
| Precision | 95.9% (71 of 74 alerts were real fraud) |
| False alarms | 3 out of 56,651 legit transactions |

So when it flags something, it's almost always right. It misses about a quarter of fraud, and I explain why further down.

<img src="docs/confusion_matrix.png" alt="Confusion matrix on the test set" width="480">

Out of 56,651 legit transactions, 56,648 went through untouched. That's 99.99%.

## How it works

The project is three separate pieces that hand off to each other:

```
 Python (ML engine)            C# (API)                     Next.js (dashboard)
 ──────────────────            ────────                     ───────────────────
 Learns what fraud      ──▶    Loads the trained      ◀──   Sends transactions one
 looks like from               model and answers            at a time and shows the
 284,807 transactions          "is this fraud?"             answers live
        │                             ▲
        └──── fraud_model.onnx ───────┘
              model_schema.json
```

The Python code never runs inside the API. Only two files cross over: the model itself (`fraud_model.onnx`) and a contract (`model_schema.json`) that says which 30 inputs the model expects, in what order, and what threshold to use. That way the API doesn't need Python installed, and the model can be swapped by dropping in two new files.

### 1. ML engine (`src/`, `notebooks/`)

Built on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 real transactions from 2 days, with features `V1`–`V28` already anonymized by the bank using PCA.

The main problem is that fraud is only 0.17% of the data. A model that says "not fraud" every single time gets 99.83% accuracy and catches a crisp zero frauds. So accuracy is useless here, and everything is measured with **PR-AUC**, **recall** and **precision** instead.

![Class imbalance](docs/class_imbalance.png)

What I found in the EDA and what I did about it:

- **Imbalance:** 1 fraud per 578 legit transactions (that's what the tiny bar above is). Handled with XGBoost's `scale_pos_weight` and a stratified split so the test set keeps the same fraud rate.
- **Duplicates:** 1,081 exact duplicate rows (19 of them fraud). Dropped before splitting, otherwise copies leak into the test set and inflate the score.
- **Time:** the raw `Time` column is "seconds since the first transaction," which means nothing for a live transaction. The transaction volume dips around 4 AM, which puts the start at about midnight, so I turned it into an **hour of day** feature. Legit volume drops at night but fraud doesn't, which is why the feature is worth keeping.
- **Scaling:** the V-features are already on a small scale from PCA, but `Amount` goes up to about 25,000, so `Amount` and `hour` needed scaling for logistic regression. XGBoost doesn't care.

![Fraud vs legit by hour of day](docs/fraud_by_hour.png)

Only 8.4% of legit transactions happen between midnight and 6 AM, but 24.3% of fraud does. That's about 3 times the share, and it's the evidence for keeping `hour`. The spikes in the fraud line are worth being careful about though: with only 2 days of data, a spike could just be one attacker running a burst.

## Experiments

### The model ladder

I compared models the way you'd climb a ladder, where each step has to beat the last one. The first step is a "model" that always says "not fraud", so anything that can't beat it hasn't learned anything.

![PR-AUC for each model](docs/model_ladder.png)

- **Logistic regression** jumps from 0.002 to 0.655 straight away. Scaling `Amount` and `hour` only adds a little (0.671), because 28 of the 30 features were already scaled by PCA. It mainly fixed a convergence warning and cut false alarms from 1,658 to 1,420.
- **XGBoost** gets to 0.822. Logistic regression can only draw one straight line between fraud and legit. XGBoost builds a few hundred small decision trees, so it can pick up combinations like "this V14 value *and* this amount".

PR-AUC is one number for the whole curve. Here's the curve itself:

![Precision-recall curves](docs/pr_curves.png)

Every point on a line is one possible threshold. Moving right catches more fraud (recall), moving down means more of the alerts are false (precision). XGBoost stays near the top much longer than logistic regression does. The dot is the threshold I actually use. After about 80% recall both lines fall off a cliff, which is the "confidently missed" fraud described further down.

### Cross-validation: class weights vs SMOTE

There are two common ways to deal with imbalance:

- **Class weights:** tell the model a missed fraud costs about 600 times more than a missed legit transaction.
- **SMOTE:** make up synthetic fraud examples by blending real ones together, until the training data is balanced.

On a single validation set, SMOTE scored 0.887 against 0.875 for class weights. That looks like a win, but the validation set only had 94 frauds, and the same model scored 0.875 on validation and 0.821 on test. A 0.05 swing from nothing but which frauds landed where. So a 0.012 gap could easily be luck.

**Cross-validation** is how you check. The training data gets split into 5 parts (folds). The model trains on 4 of them and gets scored on the 5th, and that repeats 5 times so every fold gets a turn as the test. Every one of the 378 training frauds gets scored once, and you end up with 5 scores instead of 1, so you can see how much the score moves around on its own.

![Cross-validation results](docs/cross_validation.png)

| | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± std |
|---|---|---|---|---|---|---|
| Class weights | 0.822 | 0.868 | 0.809 | 0.867 | 0.860 | 0.845 ± 0.025 |
| SMOTE | 0.839 | 0.863 | 0.801 | 0.877 | 0.865 | 0.849 ± 0.027 |

The difference between the means is 0.004. The spread inside each method is about 0.025, roughly 6 times bigger. The bands overlap almost completely, so there's no real winner on PR-AUC. I went with class weights because it gave 6 false alarms on validation against SMOTE's 19, it has fewer moving parts, and SMOTE would have to be stripped out of the pipeline before exporting to ONNX anyway.

SMOTE is also easy to get wrong: if you run it on the whole dataset before splitting, synthetic frauds end up as near-copies of real test frauds and the score lies to you. Here it runs inside an `imblearn` pipeline, so it only ever touches the training folds.

### Picking the threshold

The model outputs a probability, and 0.5 isn't special. I picked the threshold from **out-of-fold** predictions: during cross-validation, every training transaction gets predicted by the one model that didn't train on it, so all 378 training frauds can be used for tuning without touching the test set. Asking for at least 80% recall with the best precision gave **0.915**.

On the test set it gave 74.7% recall instead of 80%. That gap is within the noise cross-validation showed, and the test set just happens to have slightly harder frauds. The bigger reason recall tops out around 75% is that the frauds it misses get probabilities near zero. They look completely normal to the model, so lowering the threshold doesn't really help: even at 0.05 it only reaches about 81%.

The final model is exported to ONNX and checked against Python's predictions (max difference 9.3e-08).

## The rest of the system

### 2. API (`api/`)

An ASP.NET Core (.NET 10) minimal API that loads the ONNX model once at startup with ONNX Runtime.

| Endpoint | What it does |
|---|---|
| `GET /health` | Returns `{"status":"ok"}` |
| `GET /model-info` | Shows the loaded contract: features, threshold, version |
| `POST /predict` | Scores a transaction |

Example request:

```json
{
  "timestamp": "2026-10-10T15:47:00+03:00",
  "features": { "V1": -1.3598, "V2": -0.0728, "...": 0, "V28": -0.0211, "Amount": 149.62 }
}
```

Response:

```json
{ "fraudProbability": 0.0000173, "isFraud": false, "threshold": 0.9152014, "modelVersion": "1.0.0" }
```

The caller sends features by name, and the API puts them in the order the schema says, so nobody can mix up the order by accident. `hour` is computed from the timestamp using the transaction's own local time. If a feature is missing, it returns a 400 with the list of what's missing. If the schema or model file is broken, the API refuses to start instead of serving bad predictions.

I checked the API against Python on 495 real test transactions. The probabilities matched within 3e-08, and every fraud decision was identical.

### 3. Dashboard (`dashboard/`)

A Next.js app that replays transactions through the API and shows the results as they come in.

The transactions are real. They're rows from the test set, so the model never trained on any of them. Three things are simulated:

- **Timing:** one transaction per second (adjustable up to 10x)
- **Fraud mix:** about 5% fraud (all 95 test frauds plus 1,900 legit). The real rate is 0.17%, which means you'd wait around 10 minutes between frauds.
- **Timestamp:** each one gets the current time. I checked how much this matters: across the whole test set it changes at most 1 decision out of 56,746.

Every probability on the screen is computed live by the model through the API. Since the dashboard knows the true answer for each transaction, it also shows whether the model got it right (fraud caught, fraud missed, or false alarm) and keeps a running recall and precision.

![A caught fraud in the live feed](docs/flagged_fraud.png)

The white tick on each bar is the threshold. A bar that crosses it gets flagged.

## Try it

You don't need Python or the dataset just to run it, since the trained model and the demo transactions are already in the repo.

### With Docker (easiest)

All you need is [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```
git clone https://github.com/tseliasas/fraud-detection-engine.git
cd fraud-detection-engine
docker compose up --build
```

The first build takes a few minutes because it downloads the .NET and Node images. After that, open http://localhost:3000.

The API runs on port 5260 and the dashboard on 3000. If either port is taken on your machine, pick different ones:

```
API_PORT=5300 DASHBOARD_PORT=3100 docker compose up --build              # Mac / Linux
$env:API_PORT=5300; $env:DASHBOARD_PORT=3100; docker compose up --build  # Windows PowerShell
```

The dashboard and the API's CORS setting both follow those variables, so nothing else needs changing.

### Without Docker

You need the [.NET 10 SDK](https://dotnet.microsoft.com/download) and [Node.js](https://nodejs.org) 20 or newer.

```
git clone https://github.com/tseliasas/fraud-detection-engine.git
cd fraud-detection-engine
```

Start the API in one terminal:

```
cd api
dotnet run
```

Wait for `Now listening on: http://localhost:5259`, then start the dashboard in a second terminal:

```
cd dashboard
npm install
npm run dev
```

Open http://localhost:3000.

### What you should see

A green "API online" dot and a new transaction every second. The first flagged fraud shows up within about a minute. Set the speed to 10x to get through a few hundred and watch recall and precision settle.

Other things to try:

- Open http://localhost:5260/model-info (Docker) or http://localhost:5259/model-info (without Docker) to see what the API loaded.
- Open `api/FraudApi.http` in VS Code with the REST Client extension. It has ready-made legit, fraud and missing-feature requests.
- Stop the API while the dashboard is running. You get a red banner, and the feed picks back up by itself when the API comes back.

## Tests and CI

`api.Tests/` starts the real API in memory and checks two things: `/health` answers, and a known fraud from the test set gets flagged by `/predict`. The second one covers the whole chain (schema, model file, feature order and threshold), so if any of those break, it fails.

```
dotnet test api.Tests
```

GitHub Actions runs those tests and builds both Docker images on every push to `main`. I checked that it actually catches problems by pushing a deliberately broken test: the run went red at the test step and skipped the build.

![CI runs: green, red on a deliberately broken test, then green again](docs/ci_runs.png)

## Rebuild the model yourself

This part needs Python 3.12 and the dataset.

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and put it in `data/raw/`. It's about 150 MB, which is over GitHub's file limit, so it's not in the repo.
2. Set up the environment:

```
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Mac / Linux
pip install -r requirements.txt
```

3. Retrain and re-export:

```
cd src
python final_model.py                # trains, tunes the threshold, exports the model + schema
python export_demo_transactions.py   # rebuilds the dashboard's demo data
python make_figures.py               # regenerates every chart in docs/ plus docs/stats.json
```

`python train.py` runs all the model comparisons (baselines, logistic regression, XGBoost, SMOTE, cross-validation), and `notebooks/01_eda.ipynb` has the exploration.

## Project structure

```
fraud-detection-engine/
├── notebooks/01_eda.ipynb        exploration and the decisions it led to
├── src/
│   ├── data_prep.py              load, clean (dedupe + hour feature), stratified split
│   ├── train.py                  model comparisons and cross-validation
│   ├── final_model.py            final training, threshold tuning, ONNX export
│   ├── export_demo_transactions.py
│   └── make_figures.py           charts and numbers used in this README
├── docs/                         charts, screenshots and stats.json
├── models/
│   ├── fraud_model.onnx          the trained model
│   └── model_schema.json         the contract between Python and C#
├── api/                          C# inference API (+ Dockerfile)
├── api.Tests/                    xUnit tests for the API
├── dashboard/                    Next.js live dashboard (+ Dockerfile)
├── compose.yaml                  runs the API and dashboard together
└── .github/workflows/ci.yml      tests + Docker builds on every push
```

## Limitations

- **Only 2 days of data.** The hour-of-day pattern comes from two nights, so some of the fraud spikes might just be a single attacker.
- **Small test set.** 95 test frauds means recall moves about 1% per fraud. Cross-validation puts PR-AUC at about 0.85 ± 0.03, so treat any single number with that in mind.
- **Anonymized features.** `V1`–`V28` can't be interpreted, so I can't say *why* a transaction looks like fraud. I can only say that it does.
- **Missed frauds are confident misses.** About a quarter of fraud looks normal to the model, and no threshold fixes that. Doing better would need different features, not more tuning.

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn, XGBoost, ONNX, C# / ASP.NET Core (.NET 10), ONNX Runtime, xUnit, Next.js, React, TypeScript, Tailwind CSS, Docker, Docker Compose, GitHub Actions
