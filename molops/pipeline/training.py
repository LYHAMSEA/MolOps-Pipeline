from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import os

os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "molops-egfr-bioactivity"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Minimum performance gates -- if these are not met, CI fails
MIN_R2 = 0.50
MAX_RMSE = 1.20


# ---------------------------------------------------------------------------
# Train and track
# ---------------------------------------------------------------------------

def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[str, float, float]:
    """
    Train a Random Forest regressor and log everything to MLflow.

    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    with mlflow.start_run(run_name="random_forest_ecfp4") as run:
        params = {
            "model": "RandomForest",
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "features": "morgan_ecfp4_2048bit",
            "target": "EGFR_pIC50",
            "random_state": RANDOM_STATE,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        mlflow.log_params(params)

        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        mlflow.log_metrics({"test_rmse": rmse, "test_r2": r2})
        mlflow.sklearn.log_model(model, "random_forest")

        logger.info(
            "Random Forest -- RMSE=%.3f  R2=%.3f  run=%s",
            rmse, r2, run.info.run_id,
        )
        return run.info.run_id, rmse, r2

        model.save_model("models/xgboost.json")


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[str, float, float]:
    """
    Train XGBoost and log to MLflow.

    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    with mlflow.start_run(run_name="xgboost_ecfp4") as run:
        params = {
            "model": "XGBoost",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "features": "morgan_ecfp4_2048bit",
            "target": "EGFR_pIC50",
            "random_state": RANDOM_STATE,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        mlflow.log_params(params)

        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        mlflow.log_metrics({"test_rmse": rmse, "test_r2": r2})
        mlflow.xgboost.log_model(model, "xgboost")

        logger.info(
            "XGBoost -- RMSE=%.3f  R2=%.3f  run=%s",
            rmse, r2, run.info.run_id,
        )
        return run.info.run_id, rmse, r2


def meets_minimum_performance(rmse: float, r2: float) -> bool:
   
    return rmse <= MAX_RMSE and r2 >= MIN_R2