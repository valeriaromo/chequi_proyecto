
from __future__ import annotations

from flask import Flask, jsonify, send_from_directory, abort
from pathlib import Path

from .routes import api as api_routes
from .routes import alerts as alerts_routes
from .routes import dashboard as dashboard_routes
from .routes import docs as docs_routes
from .routes import model as model_routes
from .services import metrics as metrics_service


def create_app() -> Flask:
    app = Flask(__name__)

    # Ruta absoluta a los resultados del análisis
    BASE_DIR = Path(__file__).resolve().parent

    

    def _find_data_analysis_dir() -> Path:
        for p in Path(__file__).resolve().parents:
            for cand in (p / "data_analysis" / "output" / "data_analysis",
                        p / "src" / "data_analysis" / "output" / "data_analysis"):
                if cand.is_dir():
                    return cand
        raise FileNotFoundError("No se encontró output/data_analysis")

    DATA_ANALYSIS_DIR = _find_data_analysis_dir()
    print("DATA_ANALYSIS_DIR =", DATA_ANALYSIS_DIR)   # temporal, para verificar

    # Mantener JSON con caracteres en español
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

    # Servir figuras de los diferentes perfiles PCA
    @app.route('/data-figures/<perfil>/<filename>')
    def data_figures(perfil, filename):

        if perfil not in ("base8", "sin_aod"):
            abort(404)

        figures_dir = DATA_ANALYSIS_DIR / perfil / "figures"

        return send_from_directory(figures_dir, filename)

    return app
