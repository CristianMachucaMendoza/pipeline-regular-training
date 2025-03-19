"""
DAG: fraud_detection_training
Description: DAG for periodic training of fraud detection model with Dataproc and PySpark.
"""

import uuid
import json
import requests
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.settings import Session
from airflow.models import Connection, Variable
from airflow.utils.trigger_rule import TriggerRule
from airflow.providers.yandex.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocCreatePysparkJobOperator,
    DataprocDeleteClusterOperator
)

# Общие переменные для вашего облака
YC_ZONE = Variable.get("YC_ZONE")
YC_FOLDER_ID = Variable.get("YC_FOLDER_ID")
YC_SUBNET_ID = Variable.get("YC_SUBNET_ID")
YC_SSH_PUBLIC_KEY = Variable.get("YC_SSH_PUBLIC_KEY")

# Переменные для подключения к Object Storage
S3_ENDPOINT_URL = Variable.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY = Variable.get("S3_ACCESS_KEY")
S3_SECRET_KEY = Variable.get("S3_SECRET_KEY")
S3_BUCKET_NAME = Variable.get("S3_BUCKET_NAME")
S3_INPUT_DATA_BUCKET = f"s3a://{S3_BUCKET_NAME}/fraud_data/"  # Путь к данным
S3_OUTPUT_MODEL_BUCKET = f"s3a://{S3_BUCKET_NAME}/models/"    # Путь для сохранения моделей
S3_SRC_BUCKET = f"s3a://{S3_BUCKET_NAME}/src/"               # Путь к исходному коду
S3_DP_LOGS_BUCKET = f"s3a://{S3_BUCKET_NAME}/airflow_logs/"  # Путь для логов Data Proc

# Переменные необходимые для создания Dataproc кластера
DP_SA_AUTH_KEY_PUBLIC_KEY = Variable.get("DP_SA_AUTH_KEY_PUBLIC_KEY")
DP_SA_JSON = Variable.get("DP_SA_JSON")
DP_SA_ID = Variable.get("DP_SA_ID")
DP_SECURITY_GROUP_ID = Variable.get("DP_SECURITY_GROUP_ID")

# MLflow переменные
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = "fraud_detection"

# Создание подключения для Object Storage
YC_S3_CONNECTION = Connection(
    conn_id="yc-s3",
    conn_type="s3",
    host=S3_ENDPOINT_URL,
    extra={
        "aws_access_key_id": S3_ACCESS_KEY,
        "aws_secret_access_key": S3_SECRET_KEY,
        "host": S3_ENDPOINT_URL,
    },
)
# Создание подключения для Dataproc
YC_SA_CONNECTION = Connection(
    conn_id="yc-sa",
    conn_type="yandexcloud",
    extra={
        "extra__yandexcloud__public_ssh_key": DP_SA_AUTH_KEY_PUBLIC_KEY,
        "extra__yandexcloud__service_account_json": DP_SA_JSON,
    },
)

# Проверка наличия подключений в Airflow
def setup_airflow_connections(*connections: Connection) -> None:
    """
    Check and add missing connections to Airflow.

    Parameters
    ----------
    *connections : Connection
        Variable number of Airflow Connection objects to verify and add

    Returns
    -------
    None
    """
    session = Session()
    try:
        for conn in connections:
            print("Checking connection:", conn.conn_id)
            if not session.query(Connection).filter(Connection.conn_id == conn.conn_id).first():
                session.add(conn)
                print("Added connection:", conn.conn_id)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


# Функция для выполнения setup_airflow_connections в рамках оператора
def run_setup_connections(**kwargs): # pylint: disable=unused-argument
    """Создает подключения внутри оператора"""
    setup_airflow_connections(YC_S3_CONNECTION, YC_SA_CONNECTION)
    return True


# Функция для получения метрик последней успешной модели из MLflow
def get_best_model_metrics(**kwargs):
    """
    Получает метрики лучшей модели из MLflow

    Returns
    -------
    dict
        Словарь с метриками лучшей модели или пустой словарь, если модели нет
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Получаем ID эксперимента
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        print(f"Эксперимент {MLFLOW_EXPERIMENT_NAME} не найден. Создаем новый.")
        experiment_id = client.create_experiment(MLFLOW_EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    # Получаем все запуски для эксперимента
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.auc DESC"]
    )

    if not runs:
        print("Нет завершенных запусков для эксперимента.")
        return {}

    # Получаем лучший запуск по метрике AUC
    best_run = runs[0]
    best_metrics = {
        "run_id": best_run.info.run_id,
        "auc": best_run.data.metrics.get("auc", 0),
        "accuracy": best_run.data.metrics.get("accuracy", 0),
        "f1": best_run.data.metrics.get("f1", 0)
    }

    print(f"Лучшая модель: {best_metrics}")
    return best_metrics


# Функция для сравнения метрик новой модели с лучшей моделью
def compare_model_metrics(**kwargs):
    """
    Сравнивает метрики новой модели с лучшей моделью и решает,
    использовать ли новую модель или оставить старую

    Returns
    -------
    str
        Путь в DAG: 'use_new_model' или 'keep_current_model'
    """
    ti = kwargs['ti']

    # Получаем метрики лучшей модели
    best_model_metrics = ti.xcom_pull(task_ids='get_best_model_metrics')

    # Если нет лучшей модели, используем новую
    if not best_model_metrics:
        print("Нет предыдущей модели. Используем новую модель.")
        return 'use_new_model'

    # Получаем метрики новой модели из MLflow
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Получаем ID последнего запуска
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        print("Не удалось получить информацию о новой модели.")
        return 'keep_current_model'

    new_run = runs[0]
    new_metrics = {
        "run_id": new_run.info.run_id,
        "auc": new_run.data.metrics.get("auc", 0),
        "accuracy": new_run.data.metrics.get("accuracy", 0),
        "f1": new_run.data.metrics.get("f1", 0)
    }

    print(f"Новая модель: {new_metrics}")

    # Сравниваем метрики
    if new_metrics["auc"] > best_model_metrics["auc"]:
        print("Новая модель лучше. Используем новую модель.")
        return 'use_new_model'
    else:
        print("Текущая модель лучше. Оставляем текущую модель.")
        return 'keep_current_model'


# Функция для регистрации новой модели как production
def register_new_model(**kwargs):
    """
    Регистрирует новую модель как production в MLflow
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Получаем ID последнего запуска
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        print("Не удалось получить информацию о новой модели.")
        return

    new_run = runs[0]
    run_id = new_run.info.run_id

    # Регистрируем модель
    model_uri = f"runs:/{run_id}/model"
    model_details = mlflow.register_model(model_uri=model_uri, name="fraud_detection_model")

    # Устанавливаем тег production
    client.transition_model_version_stage(
        name="fraud_detection_model",
        version=model_details.version,
        stage="Production"
    )

    print(f"Модель {model_details.name} версии {model_details.version} зарегистрирована как Production")


# Настройки DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="fraud_detection_training",
    default_args=default_args,
    description="Periodic training of fraud detection model",
    schedule_interval=timedelta(days=1),  # Запуск раз в день
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['mlops', 'fraud_detection'],
) as dag:
    # Задача для создания подключений
    setup_connections = PythonOperator(
        task_id="setup_connections",
        python_callable=run_setup_connections,
    )

    # Получение метрик лучшей модели
    get_best_model = PythonOperator(
        task_id="get_best_model_metrics",
        python_callable=get_best_model_metrics,
    )

    # Создание Dataproc кластера
    create_cluster = DataprocCreateClusterOperator(
        task_id="create_dataproc_cluster",
        folder_id=YC_FOLDER_ID,
        cluster_name=f"fraud-detection-{uuid.uuid4()}",
        cluster_description="Temporary cluster for fraud detection model training",
        subnet_id=YC_SUBNET_ID,
        s3_bucket=S3_DP_LOGS_BUCKET,
        service_account_id=DP_SA_ID,
        ssh_public_keys=YC_SSH_PUBLIC_KEY,
        zone=YC_ZONE,
        cluster_image_version="2.0",
        masternode_resource_preset="s3-c2-m8",
        masternode_disk_type="network-ssd",
        masternode_disk_size=20,
        datanode_resource_preset="s3-c4-m16",
        datanode_disk_type="network-ssd",
        datanode_disk_size=50,
        datanode_count=2,
        computenode_count=0,
        services=["YARN", "SPARK", "HDFS", "MAPREDUCE"],
        connection_id=YC_SA_CONNECTION.conn_id,
    )

    # Запуск PySpark задания для обучения модели
    train_model = DataprocCreatePysparkJobOperator(
        task_id="train_fraud_detection_model",
        main_python_file_uri=f"{S3_SRC_BUCKET}fraud_detection_model.py",
        connection_id=YC_SA_CONNECTION.conn_id,
        args=[
            "--input", f"{S3_INPUT_DATA_BUCKET}",
            "--output", f"{S3_OUTPUT_MODEL_BUCKET}fraud_model_{datetime.now().strftime('%Y%m%d')}",
            "--model-type", "rf",
            "--tracking-uri", MLFLOW_TRACKING_URI,
            "--experiment-name", MLFLOW_EXPERIMENT_NAME
        ],
    )

    # Сравнение метрик моделей
    compare_models = BranchPythonOperator(
        task_id="compare_model_metrics",
        python_callable=compare_model_metrics,
    )

    # Использование новой модели
    use_new_model = PythonOperator(
        task_id="use_new_model",
        python_callable=register_new_model,
    )

    # Оставление текущей модели
    keep_current_model = DummyOperator(
        task_id="keep_current_model",
    )

    # Объединение путей
    join_paths = DummyOperator(
        task_id="join_paths",
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # Удаление Dataproc кластера
    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_dataproc_cluster",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Определение последовательности выполнения задач
    setup_connections >> get_best_model >> create_cluster >> train_model >> compare_models
    compare_models >> [use_new_model, keep_current_model] >> join_paths >> delete_cluster