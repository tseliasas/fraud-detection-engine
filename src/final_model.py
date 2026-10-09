from data_prep import load_data, clean, split
from train import evaluate, find_threshold
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np

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
    print("Threshold:", threshold)
    model.fit(X_train.to_numpy(), y_train)

    print("====== FINAL TEST SCORE ======")
    evaluate(model, X_test.to_numpy(), y_test, threshold)

    n_features = X_train.shape[1]
    onnx_model = onnxmltools.convert_xgboost(
        model,
        initial_types=[("input", FloatTensorType([None, n_features]))]
    )

    with open("../models/fraud_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("Saved ONNX model")

    session = rt.InferenceSession("../models/fraud_model.onnx")
    X_test_32 = X_test.to_numpy().astype(np.float32)

    onnx_outputs = session.run(None, {"input": X_test_32})
    onnx_probabilities = onnx_outputs[1][:, 1]

    python_probabilities = model.predict_proba(X_test.to_numpy())[:, 1]
    max_difference = np.abs(onnx_probabilities - python_probabilities).max()
    print("Max difference:", max_difference)
