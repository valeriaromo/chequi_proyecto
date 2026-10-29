from __future__ import annotations

from flask import Blueprint, abort, render_template

from ..services import functionality as functionality_service

bp = Blueprint("model", __name__, url_prefix="/model")

# Mapping URL slug -> template under templates/model/
_PAGES = {
    "pca": "model/pca.html",
    "xgboost": "model/xgboost.html",
    "walk-forward": "model/walk_forward.html",
    "shap": "model/shap.html",
}


@bp.get("/")
def index():
    blocks = functionality_service.all_blocks()
    return render_template("model/index.html", blocks=blocks)


@bp.get("/<slug>")
def detail(slug: str):
    template = _PAGES.get(slug)
    if template is None:
        abort(404)
    block = functionality_service.get_block(slug)
    return render_template(template, block=block)