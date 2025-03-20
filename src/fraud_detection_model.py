"""
Script: fraud_detection_model.py
Description: PySpark script for training a fraud detection model and logging to MLflow.
"""

import os
import sys
import argparse
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# pylint: disable=broad-exception-caught

def create_spark_session(s3_config=None):
    """
    Create and configure a Spark session.

    Parameters
    ----------
    s3_config : dict, optional
        Dictionary containing S3 configuration parameters
        (endpoint_url, access_key, secret_key)

    Returns
    -------
    SparkSession
        Configured Spark session
    """
    # Создаем базовый Builder
    builder = (SparkSession
        .builder
        .appName("FraudDetectionModel")
    )

    # Если передана конфигурация S3, добавляем настройки
    if s3_config and all(k in s3_config for k in ['endpoint_url', 'access_key', 'secret_key']):
        builder = (builder
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.endpoint", s3_config['endpoint_url'])
            .config("spark.hadoop.fs.s3a.access.key", s3_config['access_key'])
            .config("spark.hadoop.fs.s3a.secret.key", s3_config['secret_key'])
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
        )

    # Создаем и возвращаем сессию Spark
    return builder.getOrCreate()


def load_data(spark, input_path):
    """
    Load and prepare the fraud detection dataset.

    Parameters
    ----------
    spark : SparkSession
        Spark session
    input_path : str
        Path to the input data

    Returns
    -------
    tuple
        (train_df, test_df) - Spark DataFrames for training and testing
    """
    # Load the data
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Print schema and basic statistics
    print("Dataset Schema:")
    df.printSchema()
    print(f"Total records: {df.count()}")

    # Split the data into training and testing sets
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training set size: {train_df.count()}")
    print(f"Testing set size: {test_df.count()}")

    return train_df, test_df


def prepare_features(train_df, test_df):
    """
    Prepare features for model training.

    Parameters
    ----------
    train_df : DataFrame
        Training DataFrame
    test_df : DataFrame
        Testing DataFrame

    Returns
    -------
    tuple
        (train_df, test_df, feature_cols) - Prepared DataFrames and feature column names
    """
    # Identify feature columns (all except the target column 'fraud')
    feature_cols = [col for col in train_df.columns if col != 'fraud']

    return train_df, test_df, feature_cols


def train_model(train_df, test_df, feature_cols, model_type="rf", run_name="fraud_detection_model"):
    """
    Train a fraud detection model and log metrics to MLflow.

    Parameters
    ----------
    train_df : DataFrame
        Training DataFrame
    test_df : DataFrame
        Testing DataFrame
    feature_cols : list
        List of feature column names
    model_type : str
        Model type to train ('rf' for Random Forest, 'lr' for Logistic Regression)
    run_name : str
        Name for the MLflow run

    Returns
    -------
    tuple
        (best_model, metrics) - Best model and its performance metrics
    """
    # Create feature vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

    # Select model based on type
    if model_type == "rf":
        classifier = RandomForestClassifier(
            labelCol="fraud",
            featuresCol="features",
            numTrees=10,
            maxDepth=5
        )
        param_grid = ParamGridBuilder() \
            .addGrid(classifier.numTrees, [10, 20, 30]) \
            .addGrid(classifier.maxDepth, [5, 10, 15]) \
            .build()
    else:  # Logistic Regression
        classifier = LogisticRegression(
            labelCol="fraud",
            featuresCol="features",
            maxIter=10
        )
        param_grid = ParamGridBuilder() \
            .addGrid(classifier.regParam, [0.01, 0.1, 0.5]) \
            .addGrid(classifier.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()

    # Create pipeline
    pipeline = Pipeline(stages=[assembler, scaler, classifier])

    # Create evaluators
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="fraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="fraud",
        predictionCol="prediction",
        metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="fraud",
        predictionCol="prediction",
        metricName="f1"
    )

    # Create cross-validator
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator_auc,
        numFolds=3
    )

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Log model parameters
        mlflow.log_param("model_type", model_type)
        if model_type == "rf":
            mlflow.log_param("numTrees_options", [10, 20, 30])
            mlflow.log_param("maxDepth_options", [5, 10, 15])
        else:
            mlflow.log_param("regParam_options", [0.01, 0.1, 0.5])
            mlflow.log_param("elasticNetParam_options", [0.0, 0.5, 1.0])

        # Train the model
        print("Training model...")
        cv_model = cv.fit(train_df)
        best_model = cv_model.bestModel

        # Make predictions on test data
        print("Evaluating model...")
        predictions = best_model.transform(test_df)

        # Calculate metrics
        auc = evaluator_auc.evaluate(predictions)
        accuracy = evaluator_acc.evaluate(predictions)
        f1 = evaluator_f1.evaluate(predictions)

        # Log metrics
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

        # Log best model parameters
        if model_type == "rf":
            rf_model = best_model.stages[-1]
            mlflow.log_param("best_numTrees", rf_model.getNumTrees)
            mlflow.log_param("best_maxDepth", rf_model.getMaxDepth())
        else:
            lr_model = best_model.stages[-1]
            mlflow.log_param("best_regParam", lr_model.getRegParam())
            mlflow.log_param("best_elasticNetParam", lr_model.getElasticNetParam())

        # Log the model
        mlflow.spark.log_model(best_model, "model")

        # Print metrics
        print(f"AUC: {auc}")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")

        metrics = {
            "run_id": run_id,
            "auc": auc,
            "accuracy": accuracy,
            "f1": f1
        }

        return best_model, metrics


def save_model(model, output_path):
    """
    Save the trained model to the specified path.

    Parameters
    ----------
    model : PipelineModel
        Trained model
    output_path : str
        Path to save the model
    """
    model.write().overwrite().save(output_path)
    print(f"Model saved to: {output_path}")


def get_best_model_metrics(experiment_name):
    """
    Получает метрики лучшей модели из MLflow с алиасом 'champion'

    Parameters
    ----------
    experiment_name : str
        Имя эксперимента MLflow

    Returns
    -------
    dict
        Метрики лучшей модели или None, если модели нет
    """
    client = MlflowClient()

    # Получаем ID эксперимента
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Эксперимент '{experiment_name}' не найден")
        return None

    try:
        # Пытаемся получить модель по алиасу 'champion'
        model_name = f"{experiment_name}_model"

        # Проверяем, существует ли зарегистрированная модель
        try:
            registered_model = client.get_registered_model(model_name)
            print(f"Модель '{model_name}' зарегистрирована")
            print(f"Модель '{model_name}' имеет {len(registered_model.latest_versions)} версий")
        except Exception:
            print(f"Модель '{model_name}' еще не зарегистрирована")
            return None

        # Получаем версии модели и проверяем наличие алиаса 'champion'
        model_versions = client.get_latest_versions(model_name)
        champion_version = None

        for version in model_versions:
            # Проверяем наличие атрибута 'aliases' или используем тег
            if hasattr(version, 'aliases') and "champion" in version.aliases:
                champion_version = version
                break
            elif hasattr(version, 'tags') and version.tags.get('alias') == "champion":
                champion_version = version
                break

        if not champion_version:
            print("Модель с алиасом 'champion' не найдена")
            return None

        # Получаем Run ID чемпиона
        champion_run_id = champion_version.run_id

        # Получаем метрики из прогона
        run = client.get_run(champion_run_id)
        metrics = {
            "run_id": champion_run_id,
            "auc": run.data.metrics["auc"],
            "accuracy": run.data.metrics["accuracy"],
            "f1": run.data.metrics["f1"]
        }

        print(f"Текущая лучшая модель (champion): версия {champion_version.version}, Run ID: {champion_run_id}")
        print(f"Метрики: AUC={metrics['auc']:.4f}, Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

        return metrics
    except Exception as e:
        print(f"Ошибка при получении лучшей модели: {str(e)}")
        return None


def compare_and_register_model(new_metrics, experiment_name):
    """
    Сравнивает новую модель с лучшей в MLflow и регистрирует, если она лучше

    Parameters
    ----------
    new_metrics : dict
        Метрики новой модели
    experiment_name : str
        Имя эксперимента MLflow

    Returns
    -------
    bool
        True, если новая модель была зарегистрирована как лучшая
    """
    client = MlflowClient()

    # Получаем метрики лучшей модели
    best_metrics = get_best_model_metrics(experiment_name)

    # Имя модели
    model_name = f"{experiment_name}_model"

    # Создаем или получаем регистрированную модель
    try:
        client.get_registered_model(model_name)
        print(f"Модель '{model_name}' уже зарегистрирована")
    except Exception:
        client.create_registered_model(model_name)
        print(f"Создана новая регистрированная модель '{model_name}'")

    # Регистрируем новую модель как новую версию
    run_id = new_metrics["run_id"]
    model_uri = f"runs:/{run_id}/model"
    model_details = mlflow.register_model(model_uri, model_name)
    new_version = model_details.version

    # Решаем, должна ли новая модель стать 'champion'
    should_promote = False

    if not best_metrics:
        should_promote = True
        print("Это первая регистрируемая модель, она будет назначена как 'champion'")
    else:
        # Сравниваем на основе AUC (можно изменить критерий сравнения)
        if new_metrics["auc"] > best_metrics["auc"]:
            should_promote = True
            improvement = (new_metrics["auc"] - best_metrics["auc"]) / best_metrics["auc"] * 100
            print(f"Новая модель лучше на {improvement:.2f}% по AUC. Установка в качестве 'champion'")
        else:
            print(
                f"Новая модель не превосходит текущую 'champion' модель по AUC. "
                f"Текущий AUC: {best_metrics['auc']:.4f}, новый AUC: {new_metrics['auc']:.4f}"
            )

    # Если новая модель лучше, устанавливаем ее как 'champion'
    if should_promote:
        # Устанавливаем алиас 'champion' для новой версии
        try:
            # Проверяем доступность метода set_registered_model_alias
            if hasattr(client, 'set_registered_model_alias'):
                client.set_registered_model_alias(model_name, "champion", new_version)
            else:
                # Для старых версий MLflow используем тег
                client.set_model_version_tag(model_name, new_version, "alias", "champion")
        except Exception as e:
            print(f"Ошибка установки алиаса 'champion': {str(e)}")
            # Продолжаем выполнение и используем тег как запасной вариант
            client.set_model_version_tag(model_name, new_version, "alias", "champion")

        print(f"Версия {new_version} модели '{model_name}' установлена как 'champion'")
        return True

    # Если модель не лучше, устанавливаем алиас 'challenger'
    try:
        # Проверяем доступность метода set_registered_model_alias
        if hasattr(client, 'set_registered_model_alias'):
            client.set_registered_model_alias(model_name, "challenger", new_version)
        else:
            # Для старых версий MLflow используем тег
            client.set_model_version_tag(model_name, new_version, "alias", "challenger")
    except Exception as e:
        print(f"Ошибка установки алиаса 'challenger': {str(e)}")
        # Продолжаем выполнение и используем тег как запасной вариант
        client.set_model_version_tag(model_name, new_version, "alias", "challenger")

    print(f"Версия {new_version} модели '{model_name}' установлена как 'challenger'")
    return False


def main():
    """
    Main function to run the fraud detection model training.
    """
    parser = argparse.ArgumentParser(description="Fraud Detection Model Training")
    # Основные параметры
    parser.add_argument("--input", required=True, help="Input data path")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--model-type", choices=["rf", "lr"], default="rf", help="Model type (rf: Random Forest, lr: Logistic Regression)")

    # MLflow параметры
    parser.add_argument("--tracking-uri", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="fraud_detection", help="MLflow experiment name")
    parser.add_argument("--auto-register", action="store_true", help="Automatically register better models")
    parser.add_argument("--run-name", default=None, help="Name for the MLflow run")

    # S3 параметры
    parser.add_argument("--s3-endpoint-url", help="S3 endpoint URL")
    parser.add_argument("--s3-access-key", help="S3 access key")
    parser.add_argument("--s3-secret-key", help="S3 secret key")

    args = parser.parse_args()

    # Настраиваем S3 конфигурацию
    s3_config = None
    if args.s3_endpoint_url and args.s3_access_key and args.s3_secret_key:
        s3_config = {
            'endpoint_url': args.s3_endpoint_url,
            'access_key': args.s3_access_key,
            'secret_key': args.s3_secret_key
        }
        # Устанавливаем переменные окружения для MLflow
        os.environ['AWS_ACCESS_KEY_ID'] = args.s3_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = args.s3_secret_key
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = args.s3_endpoint_url

    # Set MLflow tracking URI if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    # Create or set the experiment
    mlflow.set_experiment(args.experiment_name)

    # Create Spark session
    spark = create_spark_session(s3_config)

    try:
        # Load and prepare data
        train_df, test_df = load_data(spark, args.input)
        train_df, test_df, feature_cols = prepare_features(train_df, test_df)

        # Generate run name if not provided
        run_name = args.run_name or f"fraud_detection_{args.model_type}_{os.path.basename(args.input)}"

        # Train the model
        model, metrics = train_model(train_df, test_df, feature_cols, args.model_type, run_name)

        # Save the model locally
        save_model(model, args.output)

        # Register model if requested
        if args.auto_register:
            print("Comparing and registering model...")
            compare_and_register_model(metrics, args.experiment_name)

        print("Training completed successfully!")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)
    finally:
        # Stop Spark session
        spark.stop()


if __name__ == "__main__":
    main()
