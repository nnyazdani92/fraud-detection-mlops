"""
This script trains a logistic regression model on the oversampled data 
and logs the model and metrics to mlflow.
"""

import os
import click
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
import mlflow


@click.command(
    help="""Given the output of the oversample_data script, train a logistic regression model
    and then train an xgboost model using optuna for hyperparameter tuning"""
)
@click.option("--oversampled-data-artifact-dir")
@click.option("--engineering-data-artifact-dir")
@click.option("--preprocessing-data-artifact-dir")
def train_model(oversampled_data_artifact_dir, engineering_data_artifact_dir, preprocessing_data_artifact_dir):
    """
    Given the output of the oversample_data script, train a logistic regression model 
    and then train an xgboost model using optuna for hyperparameter tuning
    """
    with mlflow.start_run():
        mlflow.sklearn.autolog()

        X_train = pd.read_parquet(os.path.join(
            oversampled_data_artifact_dir, "X_train_resampled-parquet-dir", "X_train_resampled.parquet"))
        y_train = pd.read_csv(os.path.join(
            oversampled_data_artifact_dir, "y_train_resampled-csv-dir", "y_train_resampled.csv"))

        X_val = pd.read_parquet(os.path.join(
            engineering_data_artifact_dir, "features", "X_val_enriched.parquet"))
        y_val = pd.read_csv(os.path.join(
            preprocessing_data_artifact_dir, "y_val-csv-dir", "y_val.csv"))

        X_test = pd.read_parquet(os.path.join(
            engineering_data_artifact_dir, "features", "X_test_enriched.parquet"))
        y_test = pd.read_csv(os.path.join(
            preprocessing_data_artifact_dir, "y_test-csv-dir", "y_test.csv"))

        X_train = pd.concat([X_train, X_val], ignore_index=True,
                            axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_val], ignore_index=True,
                            axis=0).reset_index(drop=True)

        clf = LogisticRegressionCV(
            cv=5, random_state=42, max_iter=1000, n_jobs=-1, penalty='elasticnet', solver='saga').fit(X_train, y_train.values.ravel())

        y_pred = clf.predict(X_test)

        mlflow.sklearn.log_model(
            clf,
            "model",
            input_example=X_train.iloc[0:1].astype(np.float64)
        )
        mlflow.log_metric("logistic_test_f1", f1_score(y_test, y_pred))
        mlflow.log_metric("logistic_test_precision",
                          precision_score(y_test, y_pred))
        mlflow.log_metric("logistic_test_recall", recall_score(y_test, y_pred))
        mlflow.log_metric("logistic_test_roc_auc",
                          roc_auc_score(y_test, y_pred))

        clf_report = classification_report(y_test, y_pred, labels=[0, 1], target_names=[
                                           "Not Fraud", "Fraud"], digits=6)
        mlflow.log_text(clf_report, "logistic_classification_report")

        def objective(trial):
            error = []
            params = {
                'objective': 'binary:logistic',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'eval_metric': 'logloss',
                'booster': 'gbtree',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'eta': trial.suggest_float('eta', 0.01, 0.5, log=True),
                'device': 'cuda:0'
            }
            with mlflow.start_run(nested=True):
                skf = StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=42)
                for train_index, val_index in skf.split(X_train, y_train):
                    X_tr, X_v = X_train.iloc[train_index], X_train.iloc[val_index]
                    y_tr, y_v = y_train.iloc[train_index], y_train.iloc[val_index]

                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dvalid = xgb.DMatrix(X_v, label=y_v)

                    bst = xgb.train(params, dtrain)
                    preds = bst.predict(dvalid)
                    error.append(roc_auc_score(y_v, preds))

                error = np.asarray(error)

                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", error.mean())

            return error.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

        mlflow.xgboost.autolog()

        best_params = study.best_params
        best_params["device"] = "cuda:0"
        best_params["objective"] = "binary:logistic"
        best_params["eval_metric"] = "logloss"
        best_params["booster"] = "gbtree"

        dtrain = xgb.DMatrix(X_train, label=y_train)

        bst = xgb.train(best_params, dtrain)

        dtest = xgb.DMatrix(X_test)
        preds = bst.predict(dtest)
        preds = (preds > 0.5).astype(np.int64)

        mlflow.xgboost.log_model(bst, "xgboost_model",
                                 input_example=X_train.iloc[0:1].astype(np.float64))
        mlflow.log_metric("xgb_test_f1", f1_score(y_test, preds))
        mlflow.log_metric("xgb_test_precision", precision_score(y_test, preds))
        mlflow.log_metric("xgb_test_recall", recall_score(y_test, preds))
        mlflow.log_metric("xgb_test_roc_auc", roc_auc_score(y_test, preds))

        clf_report_xgb = classification_report(
            y_test, preds, labels=[0, 1], target_names=["Not Fraud", "Fraud"], digits=6)
        mlflow.log_text(clf_report_xgb, "xgb_classification_report")


if __name__ == "__main__":
    train_model()
