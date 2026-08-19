"""
Azure Function: anomaly scoring endpoint.

Exposes an HTTP POST endpoint that receives one sensor reading (JSON)
and returns whether it's an anomaly, using the trained autoencoder.

The AnomalyScorer is loaded once at module import (cold start), then
reused across invocations while the worker stays warm.

Example request body:
{
  "sensor_2": 641.82, "sensor_3": 1589.70, "sensor_4": 1400.60, ...
}
"""

import json
import logging
import sys
from pathlib import Path

import azure.functions as func

# Make the copied model code importable
sys.path.append(str(Path(__file__).resolve().parent / "model_code"))
from score import AnomalyScorer  # noqa: E402

app = func.FunctionApp()

# Loaded once per worker (cold start), reused while warm
_scorer: AnomalyScorer | None = None


def get_scorer() -> AnomalyScorer:
    global _scorer
    if _scorer is None:
        artifacts_dir = Path(__file__).resolve().parent / "model_code" / "artifacts"
        _scorer = AnomalyScorer(artifacts_dir=artifacts_dir)
    return _scorer


@app.route(route="score", auth_level=func.AuthLevel.FUNCTION)
def score(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Anomaly scoring request received.")

    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Request body must be valid JSON."}),
            status_code=400,
            mimetype="application/json",
        )

    # Accept either a bare reading, or {"sensors": {...}} from the simulator
    reading = body.get("sensors", body)

    try:
        result = get_scorer().score(reading)
    except ValueError as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json",
        )
    except Exception as e:  # noqa: BLE001
        logging.exception("Scoring failed.")
        return func.HttpResponse(
            json.dumps({"error": "Internal scoring error.", "detail": str(e)}),
            status_code=500,
            mimetype="application/json",
        )

    return func.HttpResponse(
        json.dumps(result),
        status_code=200,
        mimetype="application/json",
    )
