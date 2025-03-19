"""
Script: fraud_detection_model.py
Description: PySpark script for training a fraud detection model and logging to MLflow.
"""

import os
import sys
import argparse
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def create_spark_session():
    """
    Create and configure a Spark session.

    Returns
    -------
    SparkSession
        Configured Spark session
    """
    return (SparkSession
            .builder
            .appName("FraudDetectionModel")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.endpoint", os.environ.get("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net"))
            .config("spark.hadoop.fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID"))
            .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY"))
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
            .getOrCreate())


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
    df = spark.read.parquet(input_path)

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
    # Identify feature columns (all except the target column 'isFraud')
    feature_cols = [col for col in train_df.columns if col != 'isFraud']

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
            labelCol="isFraud",
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
            labelCol="isFraud",
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
        labelCol="isFraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="isFraud",
        predictionCol="prediction",
        metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="isFraud",
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


def main():
    """Main function to execute the PySpark job"""
    parser = argparse.ArgumentParser(description="Fraud Detection Model Training")
    parser.add_argument("--input", required=True, help="Input data path (parquet format)")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--model-type", choices=["rf", "lr"], default="rf", help="Model type (rf: Random Forest, lr: Logistic Regression)")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="fraud_detection", help="MLflow experiment name")
    args = parser.parse_args()

    # Set MLflow tracking URI if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    # Set or create MLflow experiment
    mlflow.set_experiment(args.experiment_name)

    # Create Spark session
    spark = create_spark_session()

    try:
        # Load and prepare data
        train_df, test_df = load_data(spark, args.input)

        # Prepare features
        train_df, test_df, feature_cols = prepare_features(train_df, test_df)

        # Train model
        model, metrics = train_model(train_df, test_df, feature_cols, args.model_type)

        # Save model
        save_model(model, args.output)

        # Return success
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    finally:
        # Stop Spark session
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
