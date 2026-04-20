import os
import joblib
import mlflow
import pandas as pd
import numpy as np

mlflow.set_tracking_uri("file:///C:/mlruns")
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)


def build_preprocessor(X):

    cat_feat = X.select_dtypes(include=['object']).columns.tolist()
    num_feat = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer([
        ('num', numeric_pipeline, num_feat),
        ('cat', categorical_pipeline, cat_feat)
    ])

@mlflow.trace(name="training", span_type="FUNCTION")
def train_classifier(X_train, y_train_c, X_test, y_test_c):

    preprocess = build_preprocessor(X_train)

    clf = Pipeline([
        ('preprocess', preprocess),
        ('model', XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        ))
    ])

    mlflow.set_experiment("Placement Classifier")

    with mlflow.start_run() as run:

        clf.fit(X_train, y_train_c)
        y_pred_class = clf.predict(X_test)

        acc = accuracy_score(y_test_c, y_pred_class)
        prec = precision_score(y_test_c, y_pred_class, average="macro", zero_division=0)
        rec = recall_score(y_test_c, y_pred_class, average="macro", zero_division=0)
        f1 = f1_score(y_test_c, y_pred_class, average="macro", zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(clf, "artifacts/xgb_classifier.pkl")
        mlflow.sklearn.log_model(clf, "classifier")

    return run.info.run_id, clf

@mlflow.trace(name="training", span_type="FUNCTION")
def train_regressor(X_train, y_train_r, X_test, y_test_r):

    idx_train = y_train_r > 0
    idx_test = y_test_r > 0

    X_train_reg = X_train[idx_train]
    y_train_reg = y_train_r[idx_train]
    X_test_reg = X_test[idx_test]
    y_test_reg = y_test_r[idx_test]

    preprocess = build_preprocessor(X_train_reg)

    reg = Pipeline([
        ('preprocess', preprocess),
        ('model', LinearRegression())
    ])

    mlflow.set_experiment("Salary Regressor")

    with mlflow.start_run() as run:

        reg.fit(X_train_reg, y_train_reg)
        y_pred_reg = reg.predict(X_test_reg)

        r2 = r2_score(y_test_reg, y_pred_reg)
        mae = mean_absolute_error(y_test_reg, y_pred_reg)

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(reg, "artifacts/linear_regression.pkl")
        mlflow.sklearn.log_model(reg, "regressor")

    return run.info.run_id, reg