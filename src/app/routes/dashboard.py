from __future__ import annotations

from flask import Blueprint, render_template

from ..services import predictions as predictions_service

bp = Blueprint("dashboard", __name__)


@bp.get("/")
def index():
    municipalities = predictions_service.list_municipalities()
    snapshot = predictions_service.snapshot()
    return render_template(
        "dashboard.html",
        municipalities=municipalities,
        snapshot=snapshot,
    )