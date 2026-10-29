from __future__ import annotations
from flask import Flask, jsonify
from .routes import api as api_routes
from .routes import alerts as alerts_routes
from .routes import dashboard as dashboard_routes
from .routes import docs as docs_routes
from .routes import model as model_routes
from .services import metrics as metrics_service


def create_app() -> Flask:
    app = Flask(__name__)

    # mantener json con caracteres en español
    app.json.ensure_ascii = False
    app.json.sort_keys = False

    app.register_blueprint(dashboard_routes.bp)
    app.register_blueprint(alerts_routes.bp)
    app.register_blueprint(model_routes.bp)
    app.register_blueprint(docs_routes.bp)
    app.register_blueprint(api_routes.bp)

    @app.errorhandler(404)
    def _not_found(_err):
        return jsonify({"error": "not_found", "status": 404}), 404

    @app.errorhandler(500)
    def _server_error(_err):
        return jsonify({"error": "server_error", "status": 500}), 500

    @app.context_processor
    def _inject_metrics():
        return {"metrics": metrics_service.summary()}

    return app