from __future__ import annotations
from flask import Blueprint, jsonify, request
from ..services import alerts as alerts_service
from ..services import functionality as functionality_service
from ..services import metrics as metrics_service
from ..services import predictions as predictions_service

bp = Blueprint("api", __name__, url_prefix="/api")

# JSON endpoints. All currently unavailable data is returned from a mock file so the
# frontend can be built and reviewed today.  !!!!!!!!!11 When the model artifacts are wired in, 
# replace the service calls 

@bp.get("/predictions")
def predictions():
    municipio = request.args.get("municipio", "Aguascalientes")
    return jsonify(predictions_service.predict(municipio))


@bp.get("/alerts")
def alerts():
    return jsonify(alerts_service.current_alerts())


@bp.get("/metrics")
def metrics():
    return jsonify(metrics_service.summary())


@bp.get("/functionality")
def functionality():
    return jsonify(functionality_service.all_blocks())