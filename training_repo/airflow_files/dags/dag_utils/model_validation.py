import argparse
import pandas as pd
import numpy as np
import mlflow.pyfunc
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sqlalchemy import create_engine
import os
from .bert_model import PTDistilBertClassifier
import torch

def get_data_from_postgres(query: str, db_uri: str):
    engine = create_engine(db_uri)
    conn = engine.raw_connection()
    df = pd.read_sql(query, con=conn)
    conn.close()
    return df["text"].tolist(), df["label"].tolist()


def evaluate_model(model_path: str, db_uri: str, query: str):
    """
    Validate the model using test data from PostgreSQL.

    Args:
        model_path (str): MLflow model URI (pyfunc), e.g. "runs:/<run_id>/distilbert_sentiment"
        db_uri (str): Database URI for PostgreSQL.
        query (str): SQL query to fetch test data.

    Returns:
        None
    """
    # 1. Load test data
    X_raw, y_test = get_data_from_postgres(query, db_uri)
    y_test = list(map(int, y_test))
    input_df = pd.DataFrame({"text": X_raw})

    # 2. Load the pyfunc model
    if os.path.isfile(model_path) and model_path.endswith(".pt"):
        # chỉ có file .pt
        wrapper = PTDistilBertClassifier(3)
        state = torch.load(model_path, map_location="cpu")
        wrapper.model.load_state_dict(state)
        model = wrapper
        print(f"Loaded model from {model_path}")
    else:
        model = mlflow.pyfunc.load_model(model_path)
        print(f"Loaded PyFunc model from {model_path}")
        
    if isinstance(model, PTDistilBertClassifier):
        model.model.eval()
        probs = model.predict_proba(input_df["text"].tolist())
    else:
        probs = model.predict(input_df)
    
    # 4. Derive hard predictions
    preds = np.argmax(probs, axis=1)

    # 5. Compute metrics
    accuracy  = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="macro")
    recall    = recall_score(y_test, preds, average="macro")
    f1        = f1_score(y_test, preds, average="macro")

    # 6. Compute ROC-AUC
    if probs.shape[1] == 2:
        roc_auc = roc_auc_score(y_test, probs[:, 1])
    else:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        roc_auc = roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovr")

    # 7. Log to MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision_macro", precision)
    mlflow.log_metric("test_recall_macro", recall)
    mlflow.log_metric("test_f1_macro", f1)
    mlflow.log_metric("test_roc_auc", roc_auc)

    # 8. Print & save report
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    report = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "roc_auc": roc_auc,
    }
    pd.DataFrame([report]).to_csv("validation_report.csv", index=False)
    print("Saved validation_report.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a pyfunc MLflow model using PostgreSQL data.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="MLflow model URI, e.g. runs:/<run_id>/distilbert_sentiment")
    parser.add_argument("--db-uri", type=str, required=True, help="PostgreSQL connection URI.")
    parser.add_argument("--query", type=str, required=True, help="SQL query to fetch test data.")
    args = parser.parse_args()
    evaluate_model(args.model_path, args.db_uri, args.query)
