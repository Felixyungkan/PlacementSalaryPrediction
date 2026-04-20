from data_ingestion import ingest_data
from train import train_classifier, train_regressor
from evaluation import evaluate
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
ACCURACY_THRESHOLD = 0.3


def run_pipeline():

    print("Step 1: Data Ingestion")
    ingest_data()

    df = pd.read_csv("ingested/B.csv", sep=",")

    X = df.drop(['salary_package_lpa', 'placement_status', 'student_id'], axis=1)

    y_class = df['placement_status']
    y_reg = df['salary_package_lpa']

    x_train, x_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X,
        y_class,
        y_reg,
        test_size=0.25,
        random_state=42,
        stratify=y_class
    )

    print("Step 2: Training Classifier")
    mlflow.set_experiment("Placement Classifier")
    run_id_clf, clf_model = train_classifier(
        x_train, y_train_class, x_test, y_test_class
    )

    print("Step 3: Training Regressor")
    mlflow.set_experiment("Salary Regressor")
    run_id_reg, reg_model = train_regressor(
        x_train, y_train_reg, x_test, y_test_reg
    )

    print("Step 4: Evaluation")
    acc, f1, r2, mae = evaluate(
        x_test, y_test_class, y_test_reg, run_id_clf, run_id_reg
    )

    print(f"\n=== RESULTS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")

    if acc >= ACCURACY_THRESHOLD:
        print("\nModel approved for deployment")
    else:
        print("\nModel rejected")


if __name__ == "__main__":
    run_pipeline()