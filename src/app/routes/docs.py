from __future__ import annotations

from flask import Blueprint, render_template

bp = Blueprint("docs", __name__)


@bp.get("/docs")
def index():
    return render_template("docs.html")