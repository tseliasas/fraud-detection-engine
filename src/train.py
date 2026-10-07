from data_prep import load_data, clean, split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def evaluate(model, X_test, y_test):
    fraud_probabilities = model.predict_proba(X_test)[:,1]
    predictions         = model.predict(X_test)

    print("PR-AUC:   ", average_precision_score(y_test, fraud_probabilities))
    print("Recall:   ", recall_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

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