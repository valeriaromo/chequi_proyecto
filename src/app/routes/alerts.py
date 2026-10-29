from __future__ import annotations
from flask import Blueprint, render_template

from ..services import alerts as alerts_service

bp = Blueprint("alerts", __name__)

@bp.get("/alerts")
def index():
    payload = alerts_service.current_alerts()
    return render_template("alerts.html", payload=payload)