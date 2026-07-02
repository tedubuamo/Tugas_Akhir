import os
import sys

from groq import Groq


sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv

    _env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        print(f"✅ .env loaded dari: {_env_path}")
    else:
        print(f"ℹ️  File .env tidak ditemukan di: {_env_path}")
except ImportError:
    print("⚠️  python-dotenv tidak terinstall. Variabel .env tidak dimuat.")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
FRONTEND_PUBLIC_DIR = os.path.join(FRONTEND_DIR, "public")
MODEL_DIR_NAME = "final_bert_model_update"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_DIR_NAME)

DB_FILE = os.path.normpath(os.path.join(BASE_DIR, "my_db.db"))
if not os.path.exists(DB_FILE):
    DB_FILE = os.path.normpath(os.path.join(BASE_DIR, "..", "my_db.db"))
if not os.path.exists(DB_FILE):
    DB_FILE = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "my_db.db"))


def build_groq_client(api_key=None):
    resolved_key = (api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")).strip()
    if not resolved_key:
        print("⚠️  GROQ_API_KEY not set. Groq analysis will be skipped.")
        return None
    client = Groq(api_key=resolved_key)
    print("✅ Groq Client Initialized")
    return client


print(f"📂 Working Directory: {BASE_DIR}")
print(f"📂 SQLite DB Path: {DB_FILE}")
