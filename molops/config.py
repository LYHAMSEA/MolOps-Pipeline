from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MOLOPS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    target: str = Field(default="EGFR")
    env: str = Field(default="development")
    log_level: str = Field(default="info")
    port: int = Field(default=8001)
    secret_key: str = Field(default="dev_key_change_before_production")
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8001")
    metrics_path: str = Field(default="/metrics")

    model_path: str = Field(default="models/random_forest.joblib")
    smiles_path: str = Field(default="models/training_smiles.txt")

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        return self.env == "development"


class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    mlflow_tracking_uri: str = Field(default="./mlruns")
    mlflow_experiment_name: str = Field(default="molops-egfr-bioactivity")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_mlflow_settings() -> MLflowSettings:
    return MLflowSettings()


settings = get_settings()
mlflow_cfg = get_mlflow_settings()