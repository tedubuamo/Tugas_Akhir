import os

from flask import Flask, jsonify
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from . import state
from .config import MODEL_PATH, build_groq_client
from .routes.api import api_bp
from .routes.frontend import frontend_bp
from .services.data_access import delete_duplicate_jobs_from_db, load_jobs_from_db, load_programs_from_db, load_skills_from_db


def initialize_resources():
    state.groq_client = build_groq_client()

    state.ner_pipeline = None
    try:
        if os.path.exists(MODEL_PATH):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
            state.ner_pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
            print("✅ Model AI Loaded.")
    except Exception:
        pass

    state.VALID_SKILLS_SET = load_skills_from_db()
    print(f"✅ Skills Loaded from SQLite: {len(state.VALID_SKILLS_SET)}")

    state.VALID_PROGRAMS_SET = load_programs_from_db()
    print(f"✅ Program Studi Loaded from SQLite: {len(state.VALID_PROGRAMS_SET)}")

    removed_duplicates = delete_duplicate_jobs_from_db()
    if removed_duplicates:
        print(f"✅ Duplicate jobs cleaned during startup: {removed_duplicates}")

    state.JOB_LIST = load_jobs_from_db()
    print(f"✅ Jobs Loaded from SQLite: {len(state.JOB_LIST)}")


def create_app():
    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Max-Age"] = "86400"
        return response

    @app.errorhandler(500)
    def handle_500(error):
        resp = jsonify({"error": f"Internal server error: {str(error)}"})
        resp.status_code = 500
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.errorhandler(405)
    def handle_405(error):
        resp = jsonify({"error": "Method not allowed"})
        resp.status_code = 405
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    app.register_blueprint(frontend_bp)
    app.register_blueprint(api_bp)

    initialize_resources()
    return app
