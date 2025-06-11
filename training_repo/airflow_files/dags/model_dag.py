from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.providers.http.operators.http import HttpOperator
from airflow.providers.http.sensors.http import HttpSensor
from datetime import datetime, timedelta

from dag_utils.model_engineering import train_model
from dag_utils.model_validation import evaluate_model
from dag_utils.model_deployment import register_model, compare_models, create_model_documentation
import os
import requests

postgres_user = os.getenv("POSTGRES_USER")
postgres_password = os.getenv("POSTGRES_PASSWORD")
postgres_db = os.getenv("POSTGRES_DB")

DOCUMENTATION_PATH = "docs"

# ==== Default DAG Config ====
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# ==== DAG Definition ====
with DAG(
    dag_id="model_pipeline",
    default_args=default_args,
    description="Train, evaluate, and register DistilBERT sentiment model",
    schedule_interval=None,
    start_date=datetime(2025, 4, 1),
    catchup=False,
    tags=["model_pipeline", "distilbert"],
) as dag:

    # 1. Train model
    def _train_model(**kwargs):
        db_uri = f"postgresql+psycopg2://{postgres_user}:{postgres_password}@postgres:5432/{postgres_db}"
        result = train_model(
            model_name="distilbert_sentiment",
            db_uri=db_uri,
            query="SELECT text, label FROM ready_files"
        )
        kwargs['ti'].xcom_push(key="run_id", value=result["run_id"])
        kwargs['ti'].xcom_push(key="pyfunc_model_uri", value=result["pyfunc_model_uri"])

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        provide_context=True
    )

    #  Evaluate model 
    def _evaluate_model(**kwargs):
        pyfunc_model_uri = kwargs['ti'].xcom_pull(task_ids="train_model", key="pyfunc_model_uri")
        evaluate_model(
            model_path=pyfunc_model_uri,
            db_uri=f"postgresql+psycopg2://{postgres_user}:{postgres_password}@postgres:5432/{postgres_db}",
            query="SELECT text, label FROM test_data"
        )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model
    )

    # Compare model performance
    def _compare_models(**kwargs):
        run_id = kwargs['ti'].xcom_pull(key="run_id")
        return "register_model" if compare_models(
            new_run_id=run_id,
            experiment_name="SentimentAnalysis",
            metric_key="accuracy"
        ) else "skip_register"

    compare_model_task = BranchPythonOperator(
        task_id="compare_model",
        python_callable=_compare_models,
        provide_context=True
    )

    
    def _register_model(**kwargs):
        model_uri = kwargs['ti'].xcom_pull(key="pyfunc_model_uri")
        register_model(
            model_uri=model_uri,
            model_name="distilbert_sentiment",
            tags={"version": "auto", "source": "airflow"}
        )

    register_model_task = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
        provide_context=True
    )

    # Dummy skip task if model not better
    skip_register_task = PythonOperator(
        task_id="skip_register",
        python_callable=lambda: print("Skip register - model not better")
    )

    
    def branch_by_api_status(**kwargs):
        api_url = "http://model_api/health" 

        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                print("✅ API is up — sẽ gọi reload API.")
                return "wait_for_api"
            else:
                print(f"⚠️ API response không phải 200: {response.status_code}")
                return "skip_reload"
        except requests.exceptions.RequestException as e:
            print(f"❌ API không sẵn sàng: {e}")
            return "skip_reload"

    branch_task = BranchPythonOperator(
        task_id="branch_by_api_status",
        python_callable=branch_by_api_status,
        provide_context=True
    )

    # Dummy skip task if model not registered
    skip_reload_task = PythonOperator(
        task_id="skip_reload",
        python_callable=lambda: print("Skip reload - model not registered")
    )
    # Wait for API health before reload    
    wait_api = HttpSensor(
        task_id="wait_for_api",
        http_conn_id="sentiment_api",       
        endpoint="health",
        response_check=lambda response: response.json().get("status") == "ok",
        poke_interval=10,
        timeout=60,
    )    
    # Trigger reload-model endpoint    
    reload_api = HttpOperator(
        task_id="reload_model_api",
        http_conn_id="sentiment_api",
        method="POST",
        endpoint="reload-model",
        headers={"Content-Type": "application/json"},
        response_check=lambda response: response.status_code == 200,
        log_response=True,
    )   

    # Create model documentation
    def _create_model_doc(**kwargs):
        run_id = kwargs['ti'].xcom_pull(task_ids="train_model", key="run_id")
        create_model_documentation(
            run_id=run_id,
            model_name="distilbert_sentiment",
            doc_dir=DOCUMENTATION_PATH
        )

    create_model_doc_task = PythonOperator(
        task_id="create_model_documentation",
        python_callable=_create_model_doc,
        provide_context=True
    )

    # ==== Flow ==== #
    train_model_task >> evaluate_model_task >> compare_model_task
    compare_model_task >> register_model_task >> create_model_doc_task >> branch_task
    branch_task >> wait_api >> reload_api
    branch_task >> skip_reload_task
    compare_model_task >> skip_register_task


