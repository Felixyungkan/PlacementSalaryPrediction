import mlflow
import mlflow.sklearn
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

from pathlib import Path

BASE_DIR = Path(__file__).parent


def evaluate(x_test, y_test_class, y_test_reg, run_id_clf, run_id_reg):

    clf = joblib.load("artifacts/xgb_classifier.pkl")
    reg = joblib.load("artifacts/linear_regression.pkl")

    y_pred_class = clf.predict(x_test)
    
    idx_test = y_test_reg > 0
    y_test_reg_filtered = y_test_reg[idx_test]
    X_test_filtered = x_test[idx_test]
    y_pred_reg = reg.predict(X_test_filtered)

    acc = accuracy_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class, average="macro", zero_division=0)
    r2 = r2_score(y_test_reg_filtered, y_pred_reg)
    mae = mean_absolute_error(y_test_reg_filtered, y_pred_reg)

    mlflow.set_experiment("Placement Classifier")
    with mlflow.start_run(run_id=run_id_clf):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

    mlflow.set_experiment("Salary Regressor")
    with mlflow.start_run(run_id=run_id_reg):
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

    return acc, f1, r2, mae