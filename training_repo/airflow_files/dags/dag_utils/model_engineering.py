
import os
import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import pandas as pd
from sqlalchemy import create_engine
from .bert_model import PTDistilBertClassifier, DistilBertPyFunc


def get_data_from_postgres(query: str, db_uri: str):
    engine = create_engine(db_uri)
    conn = engine.raw_connection()
    df = pd.read_sql(query, con=conn)
    conn.close()
    return df["text"].tolist(), df["label"].tolist()

def train_model(model_name: str, db_uri: str, query: str):
    # 1. Lấy data
    x, y = get_data_from_postgres(query, db_uri)

    # 2. Tạo experiment và start run
    experiment_name = "SentimentAnalysis"
    try:
        mlflow.create_experiment(experiment_name)  
    except mlflow.exceptions.MlflowException:
        pass  
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    with mlflow.start_run() as run:
        # 3. Train
        model = PTDistilBertClassifier(num_classes=3)
        print(f"Model is running on device: {model.device}")
        model.fit(
            x, y,
            epochs=1,
            lr=2e-5,
            batch_size=4,
            val_split=0.8,
            model_save_path="saved_model"  
        )

        metrics = model.evaluate(x, y)
        print(f"Model metrics: {metrics}")
        mlflow.log_metrics(metrics)

        # 4. Log params/metrics tuỳ thích
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", 1)
        mlflow.log_param("lr", 2e-5)

        mlflow.pytorch.log_model(
            pytorch_model=model.model,
            artifact_path=f"{model_name}_pytorch"
        )

        mlflow.pyfunc.log_model(
            artifact_path=f"{model_name}_pyfunc",
            python_model=DistilBertPyFunc(),
            artifacts={"model_weights": os.path.join("saved_model", "model.pt")},
            model_config={"num_classes": 3}
        )

    # Trả về run_id & các URI nếu cần
    run_id = run.info.run_id
    # pytorch_uri = f"runs:/{run_id}/{model_name}_pytorch"
    pyfunc_uri  = f"runs:/{run_id}/{model_name}_pyfunc"
    print(f"Model saved to: {pyfunc_uri}")
    return {"run_id": run_id, "pyfunc_model_uri": pyfunc_uri}


