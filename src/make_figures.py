import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from data_prep import load_data, clean, split

warnings.filterwarnings("ignore")

OUT_DIR = "../docs"
SCHEMA_PATH = "../models/model_schema.json"
SEED = 42

# Colors from a colorblind-checked palette: blue = main series, orange = comparison
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#a3a29c"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "font.size": 10.5,
    "legend.frameon": False,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})

# Save a figure into docs/
def save(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}")
    plt.close(fig)
    print("saved", name)

# Chart 1: how rare fraud is
def plot_imbalance(df):
    counts = df["Class"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.barh(["Fraud", "Legit"], [counts[1], counts[0]], color=[ORANGE, BLUE], height=0.55)
    ax.set_title("Fraud is 0.17% of transactions")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for y, value in enumerate([counts[1], counts[0]]):
        ax.text(value + 3000, y, f"{value:,}", va="center", color=INK)
    ax.set_xlim(0, counts[0] * 1.15)
    save(fig, "class_imbalance.png")
    return {"legit": int(counts[0]), "fraud": int(counts[1]), "fraud_rate": float(counts[1] / counts.sum())}

# Chart 2: legit vs fraud by hour of day
def plot_hour_pattern(df):
    legit = df[df["Class"] == 0]["hour"].value_counts(normalize=True).sort_index() * 100
    fraud = df[df["Class"] == 1]["hour"].value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(legit.index, legit.values, color=BLUE, linewidth=2, label="Legit")
    ax.plot(fraud.index, fraud.values, color=ORANGE, linewidth=2, label="Fraud")
    ax.axvspan(0, 6, color=GRID, alpha=0.6, linewidth=0)
    ax.text(0.3, ax.get_ylim()[1] * 0.92, "night", color=INK_SOFT)
    ax.set_title("Legit spending drops at night, fraud doesn't")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("% of that class's transactions")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(loc="upper right")
    save(fig, "fraud_by_hour.png")

    night = lambda s: float(s[s.index < 6].sum())
    return {"legit_share_midnight_to_6am": night(legit), "fraud_share_midnight_to_6am": night(fraud)}

# Chart 3: PR-AUC for each model on the test set
def plot_ladder(scores):
    names = list(scores.keys())
    values = list(scores.values())
    colors = [BLUE if n.startswith("XGBoost") else GRAY for n in names]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.55)
    ax.set_title("Each model had to beat the last one (PR-AUC, test set)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
    for y, value in enumerate(values[::-1]):
        ax.text(value + 0.015, y, f"{value:.3f}", va="center", color=INK)
    save(fig, "model_ladder.png")

# Chart 4: precision-recall curves with the chosen threshold marked
def plot_pr_curves(y_test, xgb_probs, logreg_probs, threshold, fraud_rate):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for probs, color, label in [(xgb_probs, BLUE, "XGBoost"), (logreg_probs, ORANGE, "Logistic regression")]:
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ax.plot(recall, precision, color=color, linewidth=2, label=f"{label} (PR-AUC {ap:.3f})")

    ax.axhline(fraud_rate, color=GRAY, linestyle="--", linewidth=1.2, label=f"Random guessing ({fraud_rate:.4f})")

    flagged = xgb_probs >= threshold
    point_recall = float((flagged & (y_test == 1)).sum() / (y_test == 1).sum())
    point_precision = float((flagged & (y_test == 1)).sum() / flagged.sum())
    ax.plot(point_recall, point_precision, "o", markersize=9, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"threshold {threshold:.3f}\nrecall {point_recall:.1%}, precision {point_precision:.1%}",
                (point_recall, point_precision), xytext=(0.3, 0.45), textcoords="data",
                color=INK, arrowprops={"arrowstyle": "-", "color": INK_SOFT})

    ax.set_title("Precision vs recall at every threshold (test set)")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that were fraud)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    save(fig, "pr_curves.png")
    return {"recall_at_threshold": point_recall, "precision_at_threshold": point_precision}

# Chart 5: 5-fold cross-validation, class weights vs SMOTE
def plot_cv(cv_scores):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for y, (name, scores) in enumerate(cv_scores.items()):
        color = BLUE if y == 0 else ORANGE
        mean, std = scores.mean(), scores.std()
        ax.fill_betweenx([y - 0.22, y + 0.22], mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot([mean, mean], [y - 0.22, y + 0.22], color=color, linewidth=2)
        ax.plot(scores, [y] * len(scores), "o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(mean + std + 0.004, y, f"{mean:.3f} ± {std:.3f}", va="center", color=INK)

    ax.set_yticks(range(len(cv_scores)), list(cv_scores.keys()))
    low = min(s.min() for s in cv_scores.values())
    high = max(s.max() for s in cv_scores.values())
    ax.set_xlim(low - 0.01, high + 0.035)
    ax.set_ylim(-0.6, len(cv_scores) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_title("5-fold cross-validation: the gap is smaller than the noise")
    ax.set_xlabel("PR-AUC per fold (dots), mean (line) and ± 1 std (band)")
    save(fig, "cross_validation.png")

# Chart 6: confusion matrix at the chosen threshold
def plot_confusion(y_test, xgb_probs, threshold):
    matrix = confusion_matrix(y_test, (xgb_probs >= threshold).astype(int))
    row_share = matrix / matrix.sum(axis=1, keepdims=True)
    labels = [["Correctly cleared", "False alarms"], ["Missed fraud", "Caught fraud"]]

    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.imshow(row_share, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blues", [SURFACE, BLUE]), vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ink = SURFACE if row_share[i, j] > 0.6 else INK
            ax.text(j, i, f"{matrix[i, j]:,}\n{labels[i][j]}\n{row_share[i, j]:.2%} of row",
                    ha="center", va="center", color=ink, fontsize=10.5)
    ax.set_xticks([0, 1], ["Predicted legit", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actually legit", "Actually fraud"])
    ax.set_title(f"Test set at threshold {threshold:.3f}")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "confusion_matrix.png")
    tn, fp, fn, tp = matrix.ravel()
    return {"true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn), "true_positives": int(tp)}

if __name__ == "__main__":
    df = clean(load_data("../data/raw/creditcard.csv"))
    X_train, X_test, y_train, y_test = split(df, 0.2, SEED)
    threshold = json.load(open(SCHEMA_PATH))["threshold"]

    stats = {"dataset": plot_imbalance(df)}
    stats["hour"] = plot_hour_pattern(df)

    # Same models as train.py / final_model.py
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    preprocessor = ColumnTransformer([("scale", StandardScaler(), ["Amount", "hour"])], remainder="passthrough")
    models = {
        "Always 'not fraud'": DummyClassifier(strategy="most_frequent"),
        "Logistic regression (unscaled)": LogisticRegression(class_weight="balanced"),
        "Logistic regression (scaled)": Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(class_weight="balanced"))]),
        "XGBoost (class weights)": XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=SEED),
    }

    test_probs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_probs[name] = model.predict_proba(X_test)[:, 1]
    ladder = {name: float(average_precision_score(y_test, p)) for name, p in test_probs.items()}
    plot_ladder(ladder)
    stats["pr_auc_test"] = ladder

    xgb_probs = test_probs["XGBoost (class weights)"]
    stats["final_model"] = plot_pr_curves(y_test.to_numpy(), xgb_probs, test_probs["Logistic regression (scaled)"],
                                          threshold, stats["dataset"]["fraud_rate"])
    stats["final_model"]["threshold"] = threshold
    stats["final_model"].update(plot_confusion(y_test, xgb_probs, threshold))

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_models = {
        "Class weights": XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=SEED),
        "SMOTE": ImbPipeline([("smote", SMOTE(random_state=SEED)), ("model", XGBClassifier(random_state=SEED))]),
    }
    cv_scores = {name: cross_val_score(m, X_train, y_train, cv=folds, scoring="average_precision")
                 for name, m in cv_models.items()}
    plot_cv(cv_scores)
    stats["cross_validation"] = {name: {"folds": s.round(4).tolist(), "mean": float(s.mean()), "std": float(s.std())}
                                 for name, s in cv_scores.items()}

    with open(f"{OUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
