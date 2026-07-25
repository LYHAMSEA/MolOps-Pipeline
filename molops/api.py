"""
MolOps Inference API.

Endpoints:
  GET  /healthz    -- Docker/Kubernetes liveness probe
  GET  /readyz     -- Kubernetes readiness probe (checks model is loaded)
  GET  /metrics    -- Prometheus scrape endpoint
  POST /predict    -- main endpoint: SMILES in, predicted pIC50 out
  GET  /docs       -- auto-generated Swagger UI (FastAPI built-in)
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

PREDICTIONS_TOTAL = Counter(
    "molops_predictions_total",
    "Total prediction requests",
    ["model", "outcome"],
)
PREDICTION_LATENCY = Histogram(
    "molops_prediction_latency_seconds",
    "Time taken per prediction",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
CONFIDENCE_GAUGE = Gauge(
    "molops_last_confidence_score",
    "Confidence score of the most recent prediction",
    ["model"],
)
TANIMOTO_GAUGE = Gauge(
    "molops_last_max_tanimoto",
    "Max Tanimoto similarity of most recent query to training set",
    ["model"],
)
OUTSIDE_AD_TOTAL = Counter(
    "molops_outside_applicability_domain_total",
    "Predictions flagged as outside the applicability domain",
)
PIC50_SUMMARY = Summary(
    "molops_predicted_pic50",
    "Distribution of predicted pIC50 values",
    ["model"],
)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_app_state: dict[str, Any] = {}

# Background task set (same RUF006 fix as ChemOps)
_background_tasks: set[Any] = set()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load the trained model and training SMILES at startup.
    This runs once when the container starts -- not on every request.
    """
    import joblib
    import os

    model_path = os.getenv("MODEL_PATH", "models/random_forest.joblib")
    smiles_path = os.getenv("SMILES_PATH", "models/training_smiles.txt")

    try:
        _app_state["model"] = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)
    except FileNotFoundError:
        logger.warning(
            "Model file not found at %s -- predictions will fail until a model is trained",
            model_path,
        )
        _app_state["model"] = None

    try:
        with open(smiles_path) as f:
            training_smiles = [line.strip() for line in f if line.strip()]
        from molops.pipeline.evaluation import compute_training_fingerprints
        _app_state["training_fps"] = compute_training_fingerprints(training_smiles)
        logger.info("Loaded %d training SMILES for applicability domain", len(training_smiles))
    except FileNotFoundError:
        logger.warning("Training SMILES file not found -- AD check will be skipped")
        _app_state["training_fps"] = []

    logger.info("MolOps API ready")
    yield
    logger.info("MolOps API shutting down")


app = FastAPI(
    title="MolOps",
    description=(
        "Cheminformatics inference API -- submit a SMILES string, "
        "receive a predicted bioactivity (pIC50) against EGFR."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    smiles: str = Field(
        ...,
        description="SMILES string of the molecule to predict",
        examples=["CC(=O)Oc1ccccc1C(=O)O"],
    )
    model_name: str = Field(default="random_forest", description="Model to use")


class DrugLikenessResult(BaseModel):
    MW: float
    LogP: float
    HBD: int
    HBA: int
    TPSA: float
    RotBonds: int
    lipinski_pass: bool


class PredictResponse(BaseModel):
    smiles: str
    pIC50_predicted: float
    drug_likeness: DrugLikenessResult
    applicability_domain: str    # "within" or "outside"
    max_tanimoto: float
    confidence: float
    model_used: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz", tags=["ops"], summary="Liveness probe")
async def healthz() -> dict[str, str]:
    """Returns ok if the process is alive. Used by Docker HEALTHCHECK."""
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"], summary="Readiness probe")
async def readyz() -> dict[str, Any]:
    """Returns ready only when the model is loaded. Used by Kubernetes."""
    model_loaded = _app_state.get("model") is not None
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {
        "status": "ready",
        "model": "random_forest",
        "training_compounds_in_ad": len(_app_state.get("training_fps", [])),
    }


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(req: PredictRequest) -> PredictResponse:

    from molops.pipeline.featurisation import (
        lipinski_pass,
        morgan_fingerprint,
        physicochemical_descriptors,
        smiles_to_mol,
    )
    from molops.pipeline.evaluation import check_applicability_domain

    model = _app_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Parse molecule
    mol = smiles_to_mol(req.smiles)
    if mol is None:
        PREDICTIONS_TOTAL.labels(model=req.model_name, outcome="error").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Invalid SMILES string: {req.smiles!r}",
        )

    start = time.perf_counter()

    # Featurise
    fp = morgan_fingerprint(mol).reshape(1, -1)
    desc = physicochemical_descriptors(mol)

    # Predict
    pic50 = float(model.predict(fp)[0])

    # Applicability domain
    training_fps = _app_state.get("training_fps", [])
    if training_fps:
        max_tan, within_ad = check_applicability_domain(req.smiles, training_fps)
    else:
        max_tan, within_ad = 1.0, True

    # Record metrics
    duration = time.perf_counter() - start
    outcome = "success" if within_ad else "outside_ad"

    PREDICTIONS_TOTAL.labels(model=req.model_name, outcome=outcome).inc()
    PREDICTION_LATENCY.labels(model=req.model_name).observe(duration)
    CONFIDENCE_GAUGE.labels(model=req.model_name).set(max_tan)
    TANIMOTO_GAUGE.labels(model=req.model_name).set(max_tan)
    PIC50_SUMMARY.labels(model=req.model_name).observe(pic50)

    if not within_ad:
        OUTSIDE_AD_TOTAL.inc()

    return PredictResponse(
        smiles=req.smiles,
        pIC50_predicted=round(pic50, 3),
        drug_likeness=DrugLikenessResult(
            MW=desc["MW"],
            LogP=desc["LogP"],
            HBD=desc["HBD"],
            HBA=desc["HBA"],
            TPSA=desc["TPSA"],
            RotBonds=desc["RotBonds"],
            lipinski_pass=lipinski_pass(desc),
        ),
        applicability_domain="within" if within_ad else "outside",
        max_tanimoto=round(max_tan, 3),
        confidence=round(max_tan, 3),
        model_used=req.model_name,
    )


@app.get("/metrics", tags=["ops"], summary="Prometheus scrape endpoint")
async def metrics() -> PlainTextResponse:
    """Prometheus scrapes this endpoint every 10 seconds."""
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )