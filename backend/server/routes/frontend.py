from flask import Blueprint, send_from_directory

from ..config import FRONTEND_DIR, FRONTEND_PUBLIC_DIR


frontend_bp = Blueprint("frontend", __name__)


def serve_frontend_page(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@frontend_bp.route("/")
def frontend_index():
    return serve_frontend_page("index.html")


@frontend_bp.route("/index.html")
def frontend_index_html():
    return serve_frontend_page("index.html")


@frontend_bp.route("/analyzer")
@frontend_bp.route("/analyzer.html")
def frontend_analyzer():
    return serve_frontend_page("analyzer.html")


@frontend_bp.route("/login")
@frontend_bp.route("/login.html")
def frontend_login():
    return serve_frontend_page("login.html")


@frontend_bp.route("/admin")
@frontend_bp.route("/admin.html")
def frontend_admin():
    return serve_frontend_page("admin.html")


@frontend_bp.route("/public/<path:filename>")
def frontend_public(filename):
    return send_from_directory(FRONTEND_PUBLIC_DIR, filename)
