from data_prep import load_data, clean, split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, recall_score, precision_score, confusion_matrix, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate(model, X_test, y_test, threshold=0.5):
    fraud_probabilities = model.predict_proba(X_test)[:,1]
    predictions         = (fraud_probabilities>=threshold).astype(int)

    print("PR-AUC:   ", average_precision_score(y_test, fraud_probabilities))
    print("Recall:   ", recall_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

def find_threshold(y_true, probabilities, min_recall):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)

    precision = precision[:-1]
    recall = recall[:-1]

    passes = recall >= min_recall

    good_precision = precision[passes]
    good_thresholds = thresholds[passes]

    best = np.argmax(good_precision)

    return good_thresholds[best]

def cv_score(name, model, X, y):
    folds  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=folds, scoring="average_precision")
    print(name, "PR-AUC:", scores.mean(), "±", scores.std())


if __name__ == "__main__":
    df = load_data("../data/raw/creditcard.csv")
    df = clean(df)
    X_train, X_test, y_train, y_test = split(df, 0.2, 42)

    print("======Dummy======")
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    evaluate(model, X_test, y_test)

    print("======Logistic Regression - unscaled======")
    model2 = LogisticRegression(class_weight="balanced")
    model2.fit(X_train, y_train)
    evaluate(model2, X_test, y_test)

    print("======Logistic Regression - scaled======")
    preprocessor = ColumnTransformer(
        transformers=[('scale', StandardScaler(), ['Amount', 'hour'])],
        remainder = 'passthrough'
    )

    model3 = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(class_weight="balanced"))
    ])

    model3.fit(X_train, y_train)
    evaluate(model3, X_test, y_test)

    print("======XGBoost======")

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

    legit = (y_tr==0).sum()
    fraud = (y_tr==1).sum()
    
    scale_pos_weight = legit/fraud

    model4 = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)

    model4.fit(X_tr, y_tr)
    evaluate(model4, X_test, y_test)

    val_probabilities = model4.predict_proba(X_val)[ :, 1 ]
    threshold = find_threshold(y_val, val_probabilities, 0.85)
    print("Chosen threshold:", threshold)

    print("======XGBoost @ tuned threshold======")
    evaluate(model4, X_test, y_test, threshold)

    print("======XGBoost (weights) - validation======")
    evaluate(model4, X_val, y_val)

    print("======XGBoost (SMOTE) - validation======")
    smote_model = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(random_state=42))
    ])

    smote_model.fit(X_tr, y_tr)
    evaluate(smote_model, X_val, y_val)

    print("======Cross-validation======")
    cv_score("XGBoost (weights)", model4, X_train, y_train)
    cv_score("XGBoost (SMOTE)  ", smote_model, X_train, y_train)

