import os
import pendulum
import pandas as pd
from datetime import datetime
from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
import logging
from dag_utils.etl_utils import preprocess_tweets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_csv_file(file_name, **kwargs):
    file_path = os.path.abspath(f"data/{file_name}")
    if os.path.exists(file_path):
        return file_path
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def get_daily_csv_file(**kwargs):
    execution_date = kwargs['ds']  
    file_path = f"data/raw_simulated/{execution_date}.csv"
    if os.path.exists(file_path):
        return file_path
    else:
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

def extract_from_csv(**kwargs):
    file_path = get_daily_csv_file(**kwargs)  
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File CSV không tồn tại: {file_path}")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].astype(str)  
    records = df.to_dict('records')
    logging.info(f"Số lượng bản ghi được push: {len(records)}")
    kwargs['ti'].xcom_push(key='extracted_data', value=records)
    print(f"Đã load dữ liệu từ: {file_path}")

def extract_from_test_csv(**kwargs):
    file_path = get_csv_file("test.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File CSV test không tồn tại: {file_path}")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].astype(str)  
    records = df.to_dict('records')
    logging.info(f"Số lượng bản ghi được push từ test: {len(records)}")
    kwargs['ti'].xcom_push(key='extracted_test_data', value=records)
    print(f"Đã load dữ liệu từ test.csv")

def transform_data(**kwargs):
    ti = kwargs['ti']
    extracted_data = ti.xcom_pull(key='extracted_data', task_ids='extract_from_csv')
    
    df = pd.DataFrame(extracted_data)

    df = preprocess_tweets(df)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/latest.csv", index=False)

    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)

    transformed_data = df.to_dict('records')
    ti.xcom_push(key='transformed_data', value=transformed_data)
    print("Transformed data:\n", df.head())

def transform_test_data(**kwargs):
    ti = kwargs['ti']
    extracted_test_data = ti.xcom_pull(key='extracted_test_data', task_ids='extract_from_test_csv')
    
    df = pd.DataFrame(extracted_test_data)

    logging.info(f"Dữ liệu đã được load từ XCom (test): {df.head()}")
    logging.info(f"Các cột trong DataFrame (test): {df.columns.tolist()}")

    df = preprocess_tweets(df)

    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)

    transformed_test_data = df.to_dict('records')
    ti.xcom_push(key='transformed_test_data', value=transformed_test_data)
    print("Transformed test data:\n", df.head())

def load_to_postgres(**kwargs):
    ti = kwargs['ti']
    transformed_data = ti.xcom_pull(key='validated_data', task_ids='validate_data')
    hook = PostgresHook(postgres_conn_id='postgres_default')
    
    create_sql = """
    CREATE TABLE IF NOT EXISTS ready_files (
        id SERIAL PRIMARY KEY,
        date TIMESTAMP,
        text TEXT,
        cleaned_text TEXT,
        label INTEGER
    );
    """
    hook.run(create_sql)

    rows = [(row['date'], row['text_en'], row['cleaned_tweets'], row['labels']) for row in transformed_data]
    hook.insert_rows(table='ready_files', rows=rows, target_fields=['date', 'text','cleaned_text', 'label'])
    print("Dữ liệu đã được load vào bảng ready_files")

def load_to_test_data_postgres(**kwargs):
    ti = kwargs['ti']
    transformed_test_data = ti.xcom_pull(key='transformed_test_data', task_ids='transform_test_data')
    hook = PostgresHook(postgres_conn_id='postgres_default')

    create_sql = """
    CREATE TABLE IF NOT EXISTS test_data (
        id SERIAL PRIMARY KEY,
        date TIMESTAMP,
        text TEXT,
        cleaned_text TEXT,
        label INTEGER
    );
    """
    hook.run(create_sql)

    rows = [(row['date'], row['text_en'], row['cleaned_tweets'], row['labels']) for row in transformed_test_data]
    hook.insert_rows(table='test_data', rows=rows, target_fields=['date', 'text','cleaned_text', 'label'])
    print("Dữ liệu đã được load vào bảng test_data")

def check_test_data_exists(**kwargs):
    hook = PostgresHook(postgres_conn_id='postgres_default')
    check_sql = "SELECT to_regclass('test_data');"
    result = hook.get_first(check_sql)
    
    if result[0] is None:
        print("Bảng test_data không tồn tại, sẽ tiến hành tạo và load dữ liệu.")
        return 'extract_from_test_csv'
    else:
        print("Bảng test_data đã tồn tại, không cần tạo mới.")
        return 'skip_ingest_test_data'


def validate_data(**kwargs):
    ti = kwargs['ti']
    records = ti.xcom_pull(task_ids='transform_data', key='transformed_data')
    df = pd.DataFrame(records)

    # 1. Null check
    if df['text_en'].isnull().any():
        logger.warning("The 'text_en' column contains null values!")
        raise ValueError("Null values found in 'text_en' column")
    if df['labels'].isnull().any():
        logger.warning("The 'labels' column contains null values!")
        raise ValueError("Null values found in 'labels' column")
    logger.info("Null check passed.")

    # 2. Empty string check
    empty_mask = df['text_en'].str.len() == 0
    if empty_mask.any():
        logger.warning(f"Found {empty_mask.sum()} empty strings in 'text_en' column!")
        raise ValueError("Empty strings found in 'text_en' column")
    logger.info("Empty string check passed.")

    # 3. Label set membership
    invalid_label_mask = ~df['labels'].isin([0, 1, 2])
    if invalid_label_mask.any():
        bad_vals = df.loc[invalid_label_mask, 'labels'].unique()
        logger.warning(f"Invalid label values detected: {bad_vals}")
        raise ValueError("Labels outside [0, 1, 2] detected")
    logger.info("Label validity check passed.")

    logger.info("All checks passed, data is valid.")

    ti.xcom_push(key='validated_data', value=df.to_dict('records'))


def check_if_sunday(**kwargs):
    execution_date = kwargs['execution_date']
    # 3 = Thursday
    print(f"Ngày hiện tại: {execution_date.weekday()}")
    if execution_date.weekday() == 2:
        return 'trigger_fine_tune'
    else:
        return 'skip_fine_tune'

with DAG(
    dag_id='data_pipeline',
    schedule_interval='@daily',  
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    description='ETL pipeline từ file CSV, xử lý và lưu vào PostgreSQL',
) as dag:

    extract = PythonOperator(
        task_id='extract_from_csv',
        python_callable=extract_from_csv,
        provide_context=True
    )

    extract_test = PythonOperator(
        task_id='extract_from_test_csv',
        python_callable=extract_from_test_csv,
        provide_context=True
    )

    transform = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
        provide_context=True
    )

    transform_test = PythonOperator(
        task_id='transform_test_data',
        python_callable=transform_test_data,
        provide_context=True
    )

    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True
    )

    load = PythonOperator(
        task_id='load_to_postgres',
        python_callable=load_to_postgres,
        provide_context=True
    )

    load_test = PythonOperator(
        task_id='load_to_test_data_postgres',
        python_callable=load_to_test_data_postgres,
        provide_context=True
    )

    check_test_data = BranchPythonOperator(
        task_id='check_test_data_exists',
        python_callable=check_test_data_exists,
        provide_context=True
    )

    check_weekday = BranchPythonOperator(
        task_id='check_if_sunday',
        python_callable=check_if_sunday,
        provide_context=True
    )

    trigger_fine_tune = TriggerDagRunOperator(
        task_id='trigger_fine_tune',
        trigger_dag_id='model_pipeline',  
        wait_for_completion=False,
    )

    skip_fine_tune = EmptyOperator(task_id='skip_fine_tune')
    skip_ingest_test_data = EmptyOperator(task_id='skip_ingest_test_data')

    extract >> transform >> validate >> load
    extract_test >> transform_test >> load_test
    check_test_data >> [extract_test, skip_ingest_test_data]
    check_weekday >> [trigger_fine_tune, skip_fine_tune]
    load >> check_weekday