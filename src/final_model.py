from data_prep import load_data, clean, split
from train import evaluate, find_threshold
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

MIN_RECALL = 0.80

if __name__ == "__main__":
    df = load_data("../data/raw/creditcard.csv")
    df = clean(df)
    X_train, X_test, y_train, y_test = split(df, 0.2, 42)

    legit = (y_train==0).sum()
    fraud = (y_train==1).sum()
        
    scale_pos_weight = legit/fraud

    model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probabilities = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]

    threshold = find_threshold(y_train, oof_probabilities, MIN_RECALL)

    model.fit(X_train, y_train)

    print("====== FINAL TEST SCORE ======")
    evaluate(model, X_test, y_test, threshold)