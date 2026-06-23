import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os

# Load .env file dari direktori yang sama dengan app.py
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        print(f"✅ .env loaded dari: {_env_path}")
    else:
        print(f"ℹ️  File .env tidak ditemukan di: {_env_path}")
except ImportError:
    print("⚠️  python-dotenv tidak terinstall. Variabel .env tidak dimuat.")

import json
import re
import sqlite3
import unicodedata
import datetime
import pdfplumber
import torch
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from groq import Groq
import difflib

try:
    from rapidfuzz import process as _rf_process, fuzz as _rf_fuzz
    _HAS_RAPIDFUZZ = True
    print("✅ rapidfuzz available — using for major matching")
except ImportError:
    _HAS_RAPIDFUZZ = False
    print("⚠️  rapidfuzz not found — falling back to difflib for major matching")

# ==============================================================================
# 1. KONFIGURASI (PORT 5002)
# ==============================================================================
app = Flask(__name__)
# CORS ditangani SEPENUHNYA lewat @app.after_request di bawah (satu sumber saja),
# agar tidak ada header Access-Control-Allow-Origin ganda yang membuat browser
# menolak respons ("Failed to fetch" walau status 200).


@app.after_request
def add_cors_headers(response):
    """Pastikan SETIAP respons membawa header CORS yang benar — termasuk respons
    POST upload. Ini mengatasi kasus 'status 200 di Network tab tapi Failed to
    fetch di JS', yang terjadi saat frontend (mis. localhost:5500) dan backend
    (127.0.0.1:5002) dianggap origin berbeda."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response


@app.errorhandler(500)
def handle_500(e):
    resp = jsonify({"error": f"Internal server error: {str(e)}"})
    resp.status_code = 500
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.errorhandler(405)
def handle_405(e):
    resp = jsonify({"error": "Method not allowed"})
    resp.status_code = 405
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
FRONTEND_PUBLIC_DIR = os.path.join(FRONTEND_DIR, "public")
MODEL_DIR_NAME = "final_bert_model_update"
DB_FILE = os.path.normpath(os.path.join(BASE_DIR, "my_db.db"))
if not os.path.exists(DB_FILE):
    DB_FILE = os.path.normpath(os.path.join(BASE_DIR, "..", "my_db.db"))
if not os.path.exists(DB_FILE):
    DB_FILE = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "my_db.db"))

# GROQ Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq Client Initialized")
else:
    print("⚠️  GROQ_API_KEY not set. Groq analysis will be skipped.")

print(f"📂 Working Directory: {BASE_DIR}")
print(f"📂 SQLite DB Path: {DB_FILE}")


def serve_frontend_page(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route('/')
def frontend_index():
    return serve_frontend_page('index.html')


@app.route('/index.html')
def frontend_index_html():
    return serve_frontend_page('index.html')


@app.route('/analyzer')
@app.route('/analyzer.html')
def frontend_analyzer():
    return serve_frontend_page('analyzer.html')


@app.route('/login')
@app.route('/login.html')
def frontend_login():
    return serve_frontend_page('login.html')


@app.route('/admin')
@app.route('/admin.html')
def frontend_admin():
    return serve_frontend_page('admin.html')


@app.route('/public/<path:filename>')
def frontend_public(filename):
    return send_from_directory(FRONTEND_PUBLIC_DIR, filename)

# ==============================================================================
# 2. LOAD RESOURCES
# ==============================================================================
model_path = os.path.join(BASE_DIR, MODEL_DIR_NAME)
ner_pipeline = None
try:
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        print("✅ Model AI Loaded.")
except: pass

def fetch_db_rows(query, params=()):
    if not os.path.exists(DB_FILE):
        print(f"⚠️ SQLite DB tidak ditemukan: {DB_FILE}")
        return []

    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        print(f"❌ Error membaca SQLite DB: {e}")
        return []


def load_programs_from_db():
    """Load all program studi names from DB into a set for fast lookup."""
    rows = fetch_db_rows("SELECT [Nama Program Studi (Inggris)] FROM program_studi")
    programs = set()
    for row in rows:
        name = row.get('Nama Program Studi (Inggris)', '')
        if name and name.strip():
            programs.add(name.strip())
    return programs


def validate_major_against_db(candidate, valid_programs, threshold=85):
    """
    Validate and correct a candidate major name against the program_studi DB.
    Returns the canonical DB name if a good match is found, otherwise returns
    the original candidate unchanged.

    Matching strategy:
      1. Exact case-insensitive match  → return DB canonical form
      2. rapidfuzz WRatio >= threshold AND meaningful token overlap → return DB name
      3. No confident match            → keep original extracted name

    The token-overlap guard prevents false positives where only a generic word
    like "Engineering" or "Science" is shared (e.g. "Informatics Engineering"
    must NOT match "Geomatics Engineering").
    """
    if not candidate or not valid_programs:
        return candidate

    candidate_stripped = candidate.strip()
    candidate_lower    = candidate_stripped.lower()

    # Generic words that alone do not indicate semantic similarity
    _GENERIC = {
        'engineering', 'science', 'sciences', 'studies', 'arts', 'education',
        'management', 'technology', 'technologies', 'systems', 'system',
        'design', 'applied', 'vocational', 'and', 'or', 'of', 'in', 'the',
        'for', 'at', 'to', 'a', 'an',
    }

    def _content_tokens(text):
        return {w.lower() for w in text.split() if w.lower() not in _GENERIC}

    # 1. Exact match
    for prog in valid_programs:
        if prog.lower() == candidate_lower:
            print(f"  ✓ Major exact DB match: '{prog}'")
            return prog

    # 2. Fuzzy match with content-token guard
    programs_list = list(valid_programs)

    if _HAS_RAPIDFUZZ:
        result = _rf_process.extractOne(candidate_stripped, programs_list, scorer=_rf_fuzz.WRatio)
        if result:
            matched_name, score, _ = result
            query_tokens   = _content_tokens(candidate_stripped)
            match_tokens   = _content_tokens(matched_name)
            meaningful_overlap = bool(query_tokens & match_tokens)
            print(f"  ✓ Major fuzzy (rapidfuzz): '{candidate_stripped}' → '{matched_name}' "
                  f"(score={score:.0f}, overlap={meaningful_overlap})")
            if score >= threshold and meaningful_overlap:
                return matched_name
            print(f"    ↳ Rejected (score<{threshold} or no content overlap), keeping: '{candidate_stripped}'")
    else:
        # difflib fallback — SequenceMatcher-based scoring, same token-overlap guard
        best_name, best_ratio = None, 0.0
        for prog in programs_list:
            ratio = difflib.SequenceMatcher(None, candidate_stripped.lower(), prog.lower()).ratio()
            if ratio > best_ratio:
                best_ratio, best_name = ratio, prog
        if best_name:
            # Convert ratio (0-1) to 0-100 scale to compare with threshold
            score_100 = best_ratio * 100
            query_tokens   = _content_tokens(candidate_stripped)
            match_tokens   = _content_tokens(best_name)
            meaningful_overlap = bool(query_tokens & match_tokens)
            print(f"  ✓ Major fuzzy (difflib): '{candidate_stripped}' → '{best_name}' "
                  f"(score={score_100:.0f}, overlap={meaningful_overlap})")
            if score_100 >= threshold and meaningful_overlap:
                return best_name
            print(f"    ↳ Rejected, keeping: '{candidate_stripped}'")

    return candidate_stripped


def normalize_skill_entry(skill_str):
    """
    Normalize skill entries coming from DB or extraction candidates.
    Returns normalized lowercase skill or None if the entry is noise.
    """
    if not skill_str:
        return None

    cleaned = str(skill_str).strip().rstrip(';').strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned_lower = cleaned.lower()

    if len(cleaned_lower) < 2:
        if cleaned_lower == 'c':
            return 'c'
        return None

    # Allow important technical tokens even when punctuation-heavy or short
    special_exact_map = {
        'c': 'c',
        'c#': 'c#',
        'c #': 'c#',
        'c sharp': 'c#',
        'c++': 'c++',
        'c ++': 'c++',
        'c/c++': 'c/c++',
        'axure rp': 'axure rp',
        'axure rp pro': 'axure rp',
    }
    if cleaned_lower in special_exact_map:
        return special_exact_map[cleaned_lower]

    # Normalize common versioned variants to their canonical skill names
    if re.match(r'^c\#(?:\s*[\d./]+)?$', cleaned_lower):
        return 'c#'
    if re.match(r'^c\+\+(?:\s*[\d./]+)?$', cleaned_lower):
        return 'c++'
    if re.match(r'^axure rp(?:\s+[\d./]+)?(?:\s+pro)?$', cleaned_lower):
        return 'axure rp'

    # Reject entries that are clearly experience/salary/placeholder metadata
    disallowed_patterns = [
        r'\bexp(?:erience)?\s*[:=]?\s*\d+',
        r'\b\d+\s*(?:\+)?\s*(?:years?|tahun|yrs?)\b',
        r'\bfresh\s*graduate\b',
        r'^[\d\s\W_]+$',
        r'^[^\w]*$',
        r'\b(?:idr|rp)\b',
    ]
    for pattern in disallowed_patterns:
        if re.search(pattern, cleaned_lower, re.IGNORECASE):
            return None

    # Reject entries that are too symbol-heavy or meaningless fragments
    alpha_count = sum(1 for c in cleaned_lower if c.isalpha())
    alnum_count = sum(1 for c in cleaned_lower if c.isalnum())
    if alnum_count == 0:
        return None
    if alpha_count == 0 and any(ch.isdigit() for ch in cleaned_lower):
        return None
    if alnum_count < len(cleaned_lower) * 0.55:
        return None

    # Reject obvious generic noise often found in raw requirement extraction
    generic_noise = {
        'to', 'in', 'on', 'at', 'by', 'of', 'or', 'and', 'the', 'a', 'an',
        'is', 'are', 'be', 'was', 'were', 'using', 'with', 'from', 'for',
        'data', 'web', 'work', 'help', 'time', 'user', 'system', 'process',
        'summary', 'project', 'projects', 'skill', 'skills', 'requirement',
        'requirements', 'experience', 'pengalaman', 'tahun', 'year', 'years',
        'computer', 'language', 'processing', 'models', 'model', 'power',
        'learning', 'history data', 'vision'
    }
    if cleaned_lower in generic_noise:
        return None

    tokens = re.findall(r'[a-z0-9#+./-]+', cleaned_lower)
    if not tokens:
        return None

    # Reject sentence fragments like "a a and continuous monitoring processes"
    if len(tokens) >= 2 and all(len(tok) == 1 for tok in tokens[:2]):
        return None

    stopwords = {
        'a', 'an', 'and', 'or', 'the', 'of', 'to', 'for', 'with', 'in', 'on',
        'by', 'from', 'using', 'use', 'used', 'based', 'related', 'including',
        'continuous', 'monitoring', 'process', 'processes', 'management',
        'documentation', 'artifact', 'artifacts', 'evaluation', 'evaluations'
    }
    tech_keywords = {
        'python', 'sql', 'nosql', 'database', 'postgresql', 'mysql', 'sqlite',
        'mongodb', 'redis', 'oracle', 'etl', 'elt', 'api', 'rest', 'graphql',
        'json', 'xml', 'flask', 'django', 'fastapi', 'java', 'javascript',
        'typescript', 'node', 'react', 'vue', 'angular', 'html', 'css',
        'tailwind', 'bootstrap', 'docker', 'kubernetes', 'linux', 'git',
        'github', 'gitlab', 'aws', 'azure', 'gcp', 'spark', 'hadoop', 'airflow',
        'tableau', 'powerbi', 'excel', 'pandas', 'numpy', 'scipy', 'pytorch',
        'tensorflow', 'keras', 'scikit', 'ml', 'ai', 'nlp', 'llm', 'bert',
        'data', 'analytics', 'analysis', 'visualization', 'computer', 'vision',
        'automation', 'testing', 'devops', 'backend', 'frontend', 'fullstack',
        'c', 'c#', 'c++', 'axure', 'bi'
    }
    if len(tokens) >= 4:
        tech_hits = sum(1 for tok in tokens if tok in tech_keywords)
        stopword_hits = sum(1 for tok in tokens if tok in stopwords)
        if tech_hits == 0:
            return None
        if stopword_hits >= max(2, len(tokens) // 2):
            return None

    return cleaned_lower


def is_relevant_technical_skill(normalized_skill):
    """
    Keep only skills that are plausibly relevant to IT, software engineering,
    analytics, AI/ML, cloud, QA, or data science.
    """
    if not normalized_skill:
        return False

    exact_allow = {
        'c', 'c#', 'c++', 'c/c++', 'sql', 'nosql', 'etl', 'elt', 'api', 'rest',
        'graphql', 'html', 'css', 'ui', 'ux', 'qa', 'bi', 'ml', 'ai', 'nlp',
        'llm', 'ner', 'yolo', 'git', 'linux', 'aws', 'gcp', 'r', 'spss',
        'figma', 'grafana', 'tableau', 'power bi', 'powerbi', 'statsmodels',
        'supabase', 'railway', 'apify', 'web scraping', 'computer vision',
        'natural language processing', 'large language models',
        'named entity recognition', 'time-series forecasting', 'groq api'
    }
    if normalized_skill in exact_allow:
        return True

    tech_keywords = {
        'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
        'node', 'flask', 'django', 'fastapi', 'spring', 'laravel', 'php',
        'docker', 'kubernetes', 'cloud', 'aws', 'azure', 'gcp', 'devops',
        'database', 'postgresql', 'mysql', 'sqlite', 'mongodb', 'redis', 'oracle',
        'sql', 'nosql', 'etl', 'elt', 'api', 'rest', 'graphql', 'json', 'xml',
        'data', 'analytics', 'analysis', 'analyst', 'visualization', 'dashboard',
        'tableau', 'powerbi', 'excel', 'pandas', 'numpy', 'scipy', 'matplotlib',
        'seaborn', 'tensorflow', 'pytorch', 'keras', 'scikit', 'sklearn',
        'machine', 'learning', 'deep', 'neural', 'model', 'models', 'bert',
        'nlp', 'llm', 'computer', 'vision', 'automation', 'testing', 'test',
        'qa', 'frontend', 'backend', 'fullstack', 'software', 'engineering',
        'algorithm', 'algorithms', 'network', 'security', 'cyber', 'scrum',
        'kanban', 'agile', 'git', 'github', 'gitlab', 'axure', 'figma',
        'grafana', 'tableau', 'power', 'bi', 'statsmodels', 'supabase',
        'railway', 'apify', 'scraping', 'forecasting', 'recognition',
        'language', 'processing', 'yolo', 'spss', 'groq'
    }

    tokens = re.findall(r'[a-z0-9#+./-]+', normalized_skill)
    return any(tok in tech_keywords for tok in tokens)


SKILL_ALIAS_MAP = {
    'powerbi': ['power bi', 'microsoft power bi', 'ms power bi'],
    'power bi': ['power bi', 'microsoft power bi', 'ms power bi'],
    'nlp': ['nlp', 'natural language processing', 'stanford nlp', 'opennlp'],
    'llm': ['llm', 'llms', 'large language model', 'large language models'],
    'llms': ['llm', 'llms', 'large language model', 'large language models'],
    'ner': ['ner', 'named entity recognition'],
    'git/github': ['git/github', 'git', 'github'],
    'github pages': ['github pages', 'github'],
    'web scraping': ['web scraping', 'web scraping/data collection'],
    'time-series forecasting': ['time-series forecasting', 'forecasting'],
    'groq api': ['groq api', 'api'],
}


def resolve_skill_against_db(skill, skill_db):
    """
    Resolve a skill candidate against DB using exact match first, then aliases,
    then fuzzy matching with token overlap for structured hard-skill items.
    """
    normalized = normalize_skill_entry(skill)
    if not normalized:
        return None

    db_skill_map = {s.lower(): s for s in skill_db}

    if normalized in db_skill_map:
        return db_skill_map[normalized]

    alias_candidates = [normalized]
    alias_candidates.extend(SKILL_ALIAS_MAP.get(normalized, []))

    for alias in alias_candidates:
        alias_norm = normalize_skill_entry(alias)
        if alias_norm and alias_norm in db_skill_map:
            return db_skill_map[alias_norm]

    def _tokens(text):
        return {t for t in re.findall(r'[a-z0-9#+./-]+', text.lower()) if len(t) >= 2}

    if _HAS_RAPIDFUZZ:
        query_tokens = _tokens(normalized)
        result = _rf_process.extractOne(normalized, list(db_skill_map.keys()), scorer=_rf_fuzz.WRatio)
        if result:
            matched_name, score, _ = result
            matched_tokens = _tokens(matched_name)
            if score >= 86 and (query_tokens & matched_tokens):
                return db_skill_map[matched_name]

    return None


def extract_structured_skill_candidates(skills_section_text):
    """
    Parse explicit hard-skill lists from the CV more aggressively than generic
    n-gram matching. This helps preserve real hard skills from comma-separated
    skill sections.
    """
    if not skills_section_text:
        return []

    text = str(skills_section_text)
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)
    text = re.sub(r'(?<=\w)\s*\n\s*(?=\w)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    patterns = [
        r'(?:hard|technical)\s+skills?\s*:\s*(.*?)(?=(?:soft\s+skills?|languages?|certifications?|honors|awards|experience|education|projects?)\b|$)',
        r'(?:skills?|expertise|tools)\s*:\s*(.*?)(?=(?:soft\s+skills?|languages?|certifications?|honors|awards|experience|education|projects?)\b|$)',
    ]

    collected = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            collected.append(match.group(1).strip())

    if not collected:
        collected = [text]

    candidates = []
    for block in collected:
        for item in re.split(r',|;|•|\||\u2022', block):
            cleaned = item.strip()
            cleaned = re.sub(r'^(?:hard|technical|soft)\s+skills?\s*:\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^languages?\s*:\s*', '', cleaned, flags=re.IGNORECASE)
            if not cleaned:
                continue
            candidates.append(cleaned)
            paren = re.match(r'(.+?)\s*\(([^)]+)\)$', cleaned)
            if paren:
                full = paren.group(1).strip()
                alias = paren.group(2).strip()
                if full:
                    candidates.append(full)
                if alias:
                    candidates.append(alias.rstrip('s'))
            if '/' in cleaned and len(cleaned) <= 40:
                for sub in cleaned.split('/'):
                    sub = sub.strip()
                    if len(sub) >= 2:
                        candidates.append(sub)

    # preserve order
    seen = set()
    ordered = []
    for item in candidates:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


def load_skills_from_db():
    rows = fetch_db_rows("SELECT skill FROM skills WHERE skill IS NOT NULL")
    cleaned = set()
    for row in rows:
        if row.get('skill'):
            skill = normalize_skill_entry(row['skill'])
            if skill and is_relevant_technical_skill(skill):
                cleaned.add(skill)
    return cleaned


def get_data_last_updated():
    """
    Return the last modification timestamp of the job database.
    This reflects admin-side inserts/deletes/updates more accurately than
    the moment a CV analysis happens.
    """
    try:
        if os.path.exists(DB_FILE):
            return datetime.datetime.fromtimestamp(
                os.path.getmtime(DB_FILE),
                tz=datetime.timezone.utc
            ).isoformat()
    except Exception as e:
        print(f"⚠️ Failed to read DB modified time: {e}")
    return None


def load_jobs_from_db(active_only=True):
    rows = fetch_db_rows(
        "SELECT rowid AS id, category_main, category_sub, company_description, company_name, salary, job_url, job_title, work_type, description_text, requirements, min_experience_years, min_gpa, required_degree, required_majors, is_active FROM jobs"
    )
    jobs = []
    for row in rows:
        raw_active = row.get('is_active')
        
        # Logika parsing yang tahan banting
        if raw_active is None:
            active = True # Default ke True jika data NULL/kosong dari database
        else:
            try:
                # Tangani kemungkinan float (1.0) lalu konversi ke int
                active = int(float(raw_active)) == 1
            except Exception:
                # Fallback untuk nilai teks seperti "True", "true", "yes", atau "1"
                active = str(raw_active).strip().lower() in ['1', 'true', 'yes', 'aktif']
                
        row['status'] = 'active' if active else 'expired'
        
        if active_only and not active:
            continue
            
        jobs.append(row)

    return jobs


def execute_db_change(query, params=()):
    if not os.path.exists(DB_FILE):
        print(f"⚠️ SQLite DB tidak ditemukan: {DB_FILE}")
        return 0
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
    except Exception as e:
        print(f"❌ Error menulis ke SQLite DB: {e}")
        return 0


def update_job_active_state(job_id, active):
    if not job_id:
        return False
    rowcount = execute_db_change(
        "UPDATE jobs SET is_active = ? WHERE rowid = ?",
        (1 if active else 0, job_id)
    )
    return rowcount > 0


def delete_job_from_db(job_id):
    if not job_id:
        return False
    rowcount = execute_db_change(
        "DELETE FROM jobs WHERE rowid = ?",
        (job_id,)
    )
    return rowcount > 0


def resolve_job_rowid(job_identifier):
    if not job_identifier:
        return None
    try:
        return int(job_identifier)
    except:
        return None


def insert_job_to_db(job):
    """Insert a single processed job dict into the jobs table.
    Returns the new rowid on success, or None on failure."""
    if not os.path.exists(DB_FILE):
        print(f"⚠️ SQLite DB tidak ditemukan: {DB_FILE}")
        return None
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.execute(
                """
                INSERT INTO jobs (
                    category_main, category_sub, company_description, company_name,
                    salary, job_url, job_title, work_type, description_text,
                    requirements, min_experience_years, min_gpa, required_degree,
                    required_majors, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.get('category_main'),
                    job.get('category_sub'),
                    job.get('company_description'),
                    job.get('company_name'),
                    job.get('salary'),
                    job.get('job_url'),
                    job.get('job_title'),
                    job.get('work_type'),
                    job.get('description_text'),
                    job.get('requirements'),
                    job.get('min_experience_years'),
                    job.get('min_gpa'),
                    job.get('required_degree'),
                    job.get('required_majors'),
                    1,  # newly uploaded jobs are active by default
                )
            )
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        print(f"❌ Error inserting job: {e}")
        return None


def clean_html_to_text(html_text):
    """Strip HTML tags and return clean lowercase text (mirrors the notebook)."""
    if not html_text:
        return ""
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(str(html_text), "html.parser").get_text(separator=" ").strip()
    except Exception:
        # Fallback: crude tag strip
        return re.sub(r"<[^>]+>", " ", str(html_text)).strip()


def process_uploaded_jobs_file(file_storage):
    """Read an uploaded CSV/Excel file, run the same extraction pipeline the
    notebook uses, and return a list of processed job dicts ready for insertion.

    Mirrors the columns produced by _new_API_fixed.ipynb so the UI, the
    notebook output, and the database all stay consistent.
    """
    import pandas as pd

    filename = (file_storage.filename or "").lower()
    print(f"\n{'='*55}")
    print(f"📂 [UPLOAD] Membaca file: '{file_storage.filename}'")
    print(f"{'='*55}")

    # --- Read into a DataFrame based on extension ---
    if filename.endswith(".csv"):
        df = pd.read_csv(file_storage)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_storage)
    else:
        # Try CSV first, then Excel, so a mislabeled file still works
        try:
            file_storage.stream.seek(0)
            df = pd.read_csv(file_storage)
        except Exception:
            file_storage.stream.seek(0)
            df = pd.read_excel(file_storage)

    if df is None or df.empty:
        print("❌ [UPLOAD] File kosong atau tidak terbaca.")
        return [], "File kosong atau tidak terbaca."

    print(f"✅ [UPLOAD] File terbaca: {len(df)} baris, {len(df.columns)} kolom")
    print(f"   Kolom ditemukan: {list(df.columns)[:8]}{'...' if len(df.columns)>8 else ''}")

    # Normalise column names so we can find them regardless of exact casing
    cols = {c.lower().strip(): c for c in df.columns}

    def col(*candidates):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    html_col = col("jobdescriptionhtml", "job_description_html", "description_html")
    desc_col = col("description", "description_text")
    title_col = col("title", "job_title")
    company_col = col("company/name", "company_name", "company")
    company_desc_col = col("company/description", "company_description")
    cat_main_col = col("classifications/0/main", "category_main")
    cat_sub_col = col("classifications/0/sub", "category_sub")
    salary_col = col("salary")
    url_col = col("jobsearchurl", "job_url", "url")
    worktype_col = col("worktypes/0", "work_type", "worktype")
    jobid_col = col("jobid", "job_id", "id")

    processed = []
    for _, row in df.iterrows():
        # Prefer the rich HTML description; fall back to plain description text
        raw_html = row.get(html_col) if html_col else None
        if raw_html and not (isinstance(raw_html, float)):
            # Fast HTML strip via regex — BeautifulSoup is 10x slower for bulk
            plain = re.sub(r'<[^>]+>', ' ', str(raw_html))
            plain = re.sub(r'\s+', ' ', plain).strip()
        elif desc_col:
            v = row.get(desc_col)
            plain = "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
        else:
            plain = ""

        # Education / experience extraction (fast regex, keep synchronous)
        edu = extract_education_requirements(plain.lower())

        def safe(colname):
            if not colname:
                return None
            v = row.get(colname)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            return v

        # Build job_url from jobId when no explicit URL column is present
        job_url = safe(url_col)
        if not job_url and jobid_col:
            jid = safe(jobid_col)
            if jid is not None:
                try:
                    job_url = f"https://id.jobstreet.com/id/job/{int(float(jid))}"
                except Exception:
                    job_url = None

        processed.append({
            "category_main": safe(cat_main_col),
            "category_sub": safe(cat_sub_col),
            "company_description": safe(company_desc_col),
            "company_name": safe(company_col),
            "salary": safe(salary_col),
            "job_url": job_url,
            "job_title": safe(title_col),
            "work_type": safe(worktype_col),
            "description_text": plain,
            # requirements left empty — filled by background skill extraction after insert
            "requirements": "",
            "min_experience_years": edu.get("min_experience_years"),
            "min_gpa": edu.get("min_gpa"),
            "required_degree": edu.get("degree"),
            "required_majors": ", ".join(edu.get("majors", [])) if edu.get("majors") else "",
        })

    print(f"✅ [UPLOAD] Parsing selesai: {len(processed)} baris siap dimasukkan ke DB")
    return processed, None


VALID_SKILLS_SET = load_skills_from_db()
print(f"✅ Skills Loaded from SQLite: {len(VALID_SKILLS_SET)}")

VALID_PROGRAMS_SET = load_programs_from_db()
print(f"✅ Program Studi Loaded from SQLite: {len(VALID_PROGRAMS_SET)}")

JOB_LIST = load_jobs_from_db()
print(f"✅ Jobs Loaded from SQLite: {len(JOB_LIST)}")

# ==============================================================================
# 3. LOGIC
# ==============================================================================
def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_name_heuristic(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    skip = ["curriculum", "vitae", "resume", "cv", "contact", "email"]
    for line in lines[:6]:
        if 3 < len(line) < 50 and not any(s in line.lower() for s in skip) and "@" not in line:
            return line.title()
    return "Kandidat"

def extract_gpa_degree(text):
    text_lower = text.lower()
    gpa = re.search(r"(?:gpa|ipk)\s*[:]?\s*(\d[.,]\d{1,2})", text_lower)
    gpa_val = gpa.group(1).replace(",", ".") if gpa else None
    degree = None
    degrees = ["master", "bachelor", "diploma", "sarjana", "s.kom","S.Tr. S.D.T", "s1", "s2", "d3"]
    for d in degrees:
        if d in text_lower:
            degree = d.title()
            break
    return gpa_val, degree

MAJOR_KEYWORDS = {
    "teknik": "Teknik",
    "informatika": "Informatika",
    "computer science": "Informatika",
    "bisnis": "Bisnis",
    "manajemen": "Manajemen",
    "ekonomi": "Ekonomi",
    "akuntansi": "Akuntansi",
    "keuangan": "Keuangan",
    "marketing": "Marketing",
    "komunikasi": "Komunikasi",
    "hukum": "Hukum",
    "kedokteran": "Kedokteran",
    "seni": "Seni",
    "sastra": "Sastra",
    "desain": "Desain",
    "engineering": "Teknik",
    "statistics": "Statistika",
    "statistika": "Statistika",
    "matematika": "Matematika",
    "data science": "Data Science",
    "it": "IT",
    "industri": "Teknik Industri",
    "logistik": "Logistik",
    "supply chain": "Supply Chain",
}

def extract_major_from_text(text):
    text_lower = text.lower()
    majors = []
    for keyword, major in MAJOR_KEYWORDS.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
            if major not in majors:
                majors.append(major)
    return majors[0] if majors else None

def extract_work_experience_section(cv_text):
    """
    Extract only the Work Experience / Job History section from CV text.
    Uses line-start matching to avoid grabbing inline occurrences of keywords.
    """
    if not cv_text:
        return ""

    text_lower = cv_text.lower()

    start_keywords = [
        "work experience", "professional experience", "employment history",
        "job history", "pengalaman kerja", "riwayat pekerjaan",
        "pengalaman", "experience", "pekerjaan",
    ]
    end_keywords = [
        "education", "pendidikan", "skills", "keahlian", "skill",
        "certification", "sertifikasi", "award", "penghargaan",
        "project", "proyek", "portfolio", "portofolio",
        "language", "bahasa", "reference", "referensi", "summary", "profil",
    ]

    start_pos = -1
    header_end = 0

    for kw in start_keywords:
        m = re.search(r'(?:^|\n)[ \t]*' + re.escape(kw) + r'[ \t]*(?:\n|:|$)',
                      text_lower, re.MULTILINE)
        if m:
            start_pos = m.start()
            header_end = m.end()
            break

    if start_pos == -1:
        return ""

    end_pos = len(cv_text)
    for kw in end_keywords:
        m = re.search(r'(?:^|\n)[ \t]*' + re.escape(kw) + r'[ \t]*(?:\n|:|$)',
                      text_lower[header_end:], re.MULTILINE)
        if m:
            pos = header_end + m.start()
            if pos < end_pos:
                end_pos = pos

    return cv_text[header_end:end_pos].strip()


def extract_experience_from_date_ranges(cv_text):
    """
    Scan work experience section for date ranges (e.g. 'Jan 2020 - Dec 2022', '2019 – Present')
    and accumulate total months to estimate years of experience.
    Returns float (years) or None if no dates found.
    """
    if not cv_text:
        return None

    MONTHS = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10,
        'november': 11, 'december': 12,
    }
    CURRENT_YEAR = 2026
    CURRENT_MONTH = 6

    def parse_date(token):
        token = token.strip().lower()
        if token in ('present', 'now', 'current', 'sekarang', 'saat ini'):
            return CURRENT_YEAR, CURRENT_MONTH
        # "Mon YYYY"
        m = re.match(r'([a-z]+)\s+(\d{4})', token)
        if m:
            mon = MONTHS.get(m.group(1))
            yr = int(m.group(2))
            if mon and 1990 <= yr <= 2030:
                return yr, mon
        # "YYYY"
        m = re.match(r'(\d{4})$', token)
        if m:
            yr = int(m.group(1))
            if 1990 <= yr <= 2030:
                return yr, 1
        return None

    # Pattern: <date> - <date>  (dash or en-dash)
    pattern = re.compile(
        r'([A-Za-z]+\s+\d{4}|\d{4})'       # start date
        r'\s*[-–—]\s*'
        r'([A-Za-z]+\s+\d{4}|\d{4}|[Pp]resent|[Ss]ekarang|[Nn]ow|[Cc]urrent)',
    )

    total_months = 0
    seen = set()
    for m in pattern.finditer(cv_text):
        start = parse_date(m.group(1))
        end   = parse_date(m.group(2))
        if not start or not end:
            continue
        sy, sm = start
        ey, em = end
        months = (ey - sy) * 12 + (em - sm)
        if months <= 0 or months > 600:
            continue
        key = (sy, sm, ey, em)
        if key in seen:
            continue
        seen.add(key)
        total_months += months

    return round(total_months / 12, 1) if total_months > 0 else None

def extract_education_section(cv_text):
    """
    Extract only the Education section from CV text to save Groq tokens.
    """
    if not cv_text:
        return ""

    text_lower = cv_text.lower()
    lines = cv_text.splitlines()
    start_keywords = [
        "education", "pendidikan", "academic background", "riwayat pendidikan",
        "educational background", "academic history", "latar belakang pendidikan",
        "qualifications", "academic qualification",
    ]
    end_keywords = [
        "experience", "pengalaman", "skills", "keahlian", "project", "proyek",
        "certification", "sertifikasi", "language", "bahasa", "work history",
        "employment", "career", "karir", "achievement", "penghargaan",
        "organization", "organisasi", "publication", "interest", "hobi",
    ]

    start_idx = -1
    end_idx = len(lines)

    def is_section_header(raw_line, keywords):
        line = raw_line.strip().lower()
        if not line:
            return False

        # Normalize bullets / punctuation so headers like "EDUCATION:", "PENDIDIKAN"
        # and "• EDUCATION" still match, while regular sentences do not.
        normalized = re.sub(r'^[\W_]+', '', line)
        normalized = re.sub(r'[\W_]+$', '', normalized)
        compact = re.sub(r'\s+', ' ', normalized).strip()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if compact == keyword_lower:
                return True
            if compact.startswith(keyword_lower + ':'):
                return True
            if compact.startswith(keyword_lower + ' -'):
                return True
        return False

    for idx, raw_line in enumerate(lines):
        if is_section_header(raw_line, start_keywords):
            start_idx = idx
            break

    if start_idx != -1:
        for idx in range(start_idx + 1, len(lines)):
            if is_section_header(lines[idx], end_keywords):
                end_idx = idx
                break

        section = "\n".join(lines[start_idx:end_idx]).strip()
        if len(section) >= 80:
            return section

    # Fallback: build a compact context from lines that look like education entries.
    education_anchor = re.compile(
        r'\b('
        r'education|pendidikan|academic|university|universitas|college|institut|institute|'
        r'politeknik|sekolah tinggi|bachelor|master|phd|sarjana|diploma|'
        r'jurusan|major|program studi|prodi|field of study|'
        r's\.?\s*[123]|d\.?\s*[1234]|b\.sc|bachelor\'?s|master\'?s'
        r')\b',
        re.IGNORECASE
    )
    windows = []
    for idx, raw_line in enumerate(lines):
        if education_anchor.search(raw_line):
            windows.append((max(0, idx - 1), min(len(lines), idx + 3)))

    if windows:
        merged = []
        for start, end in sorted(windows):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        section_lines = []
        for start, end in merged[:6]:
            section_lines.extend(lines[start:end])
        section = "\n".join(section_lines).strip()
        if len(section) >= 40:
            return section[:2500]

    # Last resort: early text only if no clear education context was found at all.
    return cv_text[:2500]

def analyze_cv_with_groq(cv_text):
    """
    Analyze CV using Groq LLM to extract work experience details.
    Only analyzes the Work Experience section.
    Returns: {"work_experiences": [...], "total_years": float, "roles": [...], "error": str or None}
    """
    if not groq_client or not cv_text:
        return {"work_experiences": [], "total_years": None, "roles": [], "error": "Groq not available"}
    
    try:
        # Extract only work experience section
        work_exp_section = extract_work_experience_section(cv_text)
        
        if not work_exp_section:
            print("⚠️  Work Experience section not found in CV")
            return {"work_experiences": [], "total_years": None, "roles": [], "error": "Work Experience section not found"}
        
        print(f"📋 Work Experience Section Extracted ({len(work_exp_section)} chars)")
        
        prompt = f"""You are a CV parser. Extract ONLY professional/paid work experience from the section below.

Work Experience Section:
{work_exp_section[:3500]}

Rules:
- INCLUDE: full-time jobs, part-time jobs, internships, contract work, freelance work at real companies/organizations.
- EXCLUDE: academic projects, personal projects, portfolio projects, course assignments, research projects, hackathons, or any entry where the "company" is clearly a project name (not a real organization).
- If an entry looks like a project (no real company, or labeled as "project"/"proyek"/"portofolio"), skip it entirely.

Return a JSON object with exactly these fields:
{{
  "work_experiences": [
    {{
      "company": "<real company or organization name>",
      "position": "<job title>",
      "start_year": <number>,
      "end_year": <number or "present">,
      "duration_years": <float>
    }}
  ],
  "total_years": <float — sum of all durations, 0 if none>,
  "roles": ["<job title 1>", "<job title 2>"]
}}

Return ONLY valid JSON, no markdown or extra text."""

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024,
            top_p=1,
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Try to parse JSON from response
        try:
            result = json.loads(response_text)
            result["role_groups"] = group_work_experiences_by_role(result.get("work_experiences", []))
            print(f"\n🔍 GROQ Analysis Result:")
            print(f"   Work Experiences Found: {len(result.get('work_experiences', []))}")
            print(f"   Total Years: {result.get('total_years')}")
            print(f"   Roles: {', '.join(result.get('roles', []))}")
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from response if it contains extra text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["role_groups"] = group_work_experiences_by_role(result.get("work_experiences", []))
                print(f"\n🔍 GROQ Analysis Result:")
                print(f"   Work Experiences Found: {len(result.get('work_experiences', []))}")
                print(f"   Total Years: {result.get('total_years')}")
                print(f"   Roles: {', '.join(result.get('roles', []))}")
                return result
            else:
                return {"work_experiences": [], "total_years": None, "roles": [], "role_groups": [], "error": "Could not parse Groq response"}
    except Exception as e:    
        print(f"❌ Groq Analysis Error: {str(e)}")
        return {"work_experiences": [], "total_years": None, "roles": [], "error": str(e)}

def analyze_education_with_groq(cv_text):
    """
    Analyze CV using Groq LLM to extract Degree, Major, and GPA.
    Automatically maps Indonesian degrees (S.ST, A.Md, dll) and translates majors to English.
    """
    if not groq_client or not cv_text:
        return {**extract_education_regex(cv_text or ""), "error": "Groq not available"}
        
    try:
        edu_section = extract_education_section(cv_text)
        print(f"🎓 Education Section Extracted ({len(edu_section)} chars)")

        # If the section is too short, Groq won't have enough signal — use full text instead
        groq_input = edu_section if len(edu_section) >= 100 else cv_text[:3000]

        prompt = f"""You are a CV parser. Extract education details from the CV text below and return ONLY valid JSON.

CV Text:
{groq_input[:3000]}

Return exactly these 3 fields:
{{
  "degree": "<highest university degree level>",
  "major": "<field of study in English>",
  "gpa": <GPA as float or null>
}}

Rules for "degree" — IGNORE high school (SMA, SMK, MA, SMP, or any secondary/vocational school). Only consider UNIVERSITY or COLLEGE level education. Map to ONE of these exact strings (case-sensitive):
- "PhD"      → S3, Doktor, Ph.D, Doctorate
- "Master"   → S2, Magister, M.Sc, M.S, M.Eng, M.Kom, M.T, Master's
- "Bachelor" → S1, D4, Sarjana, Bachelor's, B.S., B.Sc., B.E., B.Eng, B.Tech, B.A.,
               S.Kom, S.T, S.ST, S.Tr, S.Tr.T, S.Tr.Kom, S.Ds, S.E,
               Sarjana Terapan, Undergraduate, any 4-year university or polytechnic degree
- "Diploma"  → D3, D2, D1, A.Md, Ahli Madya, Associate's, Diploma
- IMPORTANT: "S.Tr." or "S. Tr." (Sarjana Terapan) is always "Bachelor".
- If a university or Politeknik name is present but no explicit degree, infer "Bachelor" as default.
- If multiple university degrees, return the HIGHEST one.
- Return null ONLY if there is absolutely no university/college education at all.

Rules for "major" — extract the FIELD OF STUDY (program studi/jurusan) from the HIGHEST university degree, NOT the institution name and NOT from high school:
- IGNORE: SMA, SMK, MA entries completely — do not extract major from these.
- WRONG: "Politeknik Elektronika Negeri Surabaya" → institution name, NOT the major.
- WRONG: "Electronics Engineering Vocational Education" → this is from an SMK/vocational school, ignore it.
- RIGHT: extract the program name written after "in", "jurusan", "program studi", or listed separately under a university.
- Translate to English title case. Examples:
  "Sains Data Terapan"       → "Applied Data Science"
  "Teknik Informatika"       → "Computer Science"
  "Sistem Informasi"         → "Information Systems"
  "Teknologi Informasi"      → "Information Technology"
  "Teknik Elektro"           → "Electrical Engineering"
  "Teknik Komputer"          → "Computer Engineering"
  "Rekayasa Perangkat Lunak" → "Software Engineering"
  "Ilmu Komputer"            → "Computer Science"
- If you cannot find a clear program/field name from a university, return null.

Rules for "gpa" — extract from the UNIVERSITY entry only (not high school grades). Return null if not found.

Return ONLY the JSON object. No markdown, no explanation."""

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
            top_p=1,
        )

        response_text = completion.choices[0].message.content.strip()
        result = json.loads(response_text)

        print(f"🔍 GROQ Education Result: Degree: {result.get('degree')}, Major: {result.get('major')}, GPA: {result.get('gpa')}")

        # Layer 2: fill any remaining nulls with regex
        regex_result = extract_education_regex(cv_text)
        if not result.get('degree') and regex_result.get('degree'):
            result['degree'] = regex_result['degree']
            print(f"  ↳ degree filled by regex: {result['degree']}")
        if not result.get('major') and regex_result.get('major'):
            result['major'] = regex_result['major']
            print(f"  ↳ major filled by regex: {result['major']}")
        if not result.get('gpa') and regex_result.get('gpa'):
            result['gpa'] = regex_result['gpa']
            print(f"  ↳ gpa filled by regex: {result['gpa']}")

        return result

    except Exception as e:
        print(f"❌ Groq Education Analysis Error: {str(e)}")
        regex_result = extract_education_regex(cv_text)
        print(f"  ↳ Using regex fallback: {regex_result}")
        return {**regex_result, "error": str(e)}

def extract_experience_years(text):
    """
    Extract years of work experience from CV text.
    Returns: float representing years of experience, or None
    """
    if not text: return None
    
    text_lower = text.lower()
    
    # Pattern 1: "X years", "X tahun", "X yrs"
    patterns = [
        r"(\d+)\s*(?:years?|tahun|yrs?)\s*(?:of|pengalaman|di)",
        r"(?:pengalaman|experience)\s*[:]?\s*(\d+)\s*(?:years?|tahun)",
        r"(\d+)\s*(?:tahun|years?)\s*(?:pengalaman|experience|kerja|work)",
        r"total\s*(?:pengalaman|experience)\s*[:]?\s*(\d+)\s*(?:years?|tahun)",
    ]
    
    years_list = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            years_list.extend([int(m) if isinstance(m, str) else m for m in matches])
    
    if years_list:
        # Return the maximum years found (most conservative estimate)
        max_years = max(years_list)
        return float(max_years)
    
    # Pattern 2: Date range extraction (From - To)
    # Look for patterns like "2020 - 2023" or "Jan 2020 - Dec 2023"
    date_pattern = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\s*(?:20|19)\d{2}\s*(?:-|to|sd|s\.d)\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\s*(?:20|19)\d{2}"
    date_matches = re.findall(date_pattern, text_lower, re.IGNORECASE)
    
    if date_matches:
        # Try to extract year ranges
        year_pattern = r"(20|19)(\d{2})"
        all_years = re.findall(year_pattern, text_lower)
        if len(all_years) >= 2:
            try:
                first_year = int(all_years[0][0] + all_years[0][1])
                last_year = int(all_years[-1][0] + all_years[-1][1])
                if last_year >= first_year:
                    return float(last_year - first_year + 1)
            except:
                pass
    
    return None


def calculate_duration_years(start_year, end_year):
    if start_year is None:
        return None
    current_year = datetime.datetime.now().year
    if isinstance(end_year, str) and end_year.strip().lower() in ["present", "now", "saat ini", "current"]:
        end_year = current_year
    try:
        start_year = int(start_year)
        end_year = int(end_year)
    except Exception:
        return None
    if end_year < start_year:
        return 0.0
    return float(end_year - start_year + 1)


def group_work_experiences_by_role(experiences):
    grouped = {}
    for exp in experiences or []:
        role = str(exp.get("position", "")).strip()
        if not role:
            continue
        normalized_role = role.lower()
        duration = exp.get("duration_years")
        if duration is None:
            duration = calculate_duration_years(exp.get("start_year"), exp.get("end_year"))
        if duration is None:
            duration = 0.0
        if normalized_role not in grouped:
            grouped[normalized_role] = {
                "role": role,
                "total_duration_years": 0.0,
                "experiences": []
            }
        grouped[normalized_role]["total_duration_years"] += float(duration)
        grouped[normalized_role]["experiences"].append({
            "company": exp.get("company"),
            "position": role,
            "start_year": exp.get("start_year"),
            "end_year": exp.get("end_year"),
            "duration_years": float(duration)
        })

    for group in grouped.values():
        group["total_duration_years"] = round(group["total_duration_years"], 1)
    return list(grouped.values())


def normalize_skill_variants(skill_str):
    """
    Normalize skill variants to their canonical form.
    E.g., "Delivered", "Delivering", "Delivery" → "Delivery"
    """
    if not skill_str:
        return skill_str
    
    skill_lower = skill_str.lower().strip()
    
    # Mapping dari variant ke bentuk canonical
    skill_normalization = {
        # Deliver variants
        'deliver': 'delivery', 'delivered': 'delivery', 'delivering': 'delivery',
        
        # Test variants
        'test': 'testing', 'tested': 'testing', 'tests': 'testing',
        
        # Manage variants
        'manage': 'management', 'managed': 'management', 'managing': 'management', 'manager': 'management',
        
        # Develop variants
        'develop': 'development', 'developed': 'development', 'developing': 'development', 'developer': 'development',
        
        # Design variants
        'design': 'design', 'designed': 'design', 'designing': 'design', 'designer': 'design',
        
        # Analyze variants
        'analyze': 'analysis', 'analyzed': 'analysis', 'analyzing': 'analysis', 'analyst': 'analysis',
        
        # Optimize variants
        'optimize': 'optimization', 'optimized': 'optimization', 'optimizing': 'optimization',
        
        # Implement variants
        'implement': 'implementation', 'implemented': 'implementation', 'implementing': 'implementation',
        
        # Monitor variants
        'monitor': 'monitoring', 'monitored': 'monitoring', 'monitoring': 'monitoring',
        
        # Report variants
        'report': 'reporting', 'reported': 'reporting', 'reporting': 'reporting',
        
        # Document variants
        'document': 'documentation', 'documented': 'documentation', 'documenting': 'documentation',
        
        # Visualize variants
        'visualize': 'visualization', 'visualized': 'visualization', 'visualizing': 'visualization',
        
        # Support variants
        'support': 'support', 'supported': 'support', 'supporting': 'support',
        
        # Lead variants
        'lead': 'leadership', 'leading': 'leadership', 'led': 'leadership',
        
        # Train variants
        'train': 'training', 'trained': 'training', 'training': 'training',
        
        # Plan variants
        'plan': 'planning', 'planned': 'planning', 'planning': 'planning',
        
        # Improve variants
        'improve': 'improvement', 'improved': 'improvement', 'improving': 'improvement',
        
        # Integrate variants
        'integrate': 'integration', 'integrated': 'integration', 'integrating': 'integration',
        
        # Deploy variants
        'deploy': 'deployment', 'deployed': 'deployment', 'deploying': 'deployment',
        
        # Automate variants
        'automate': 'automation', 'automated': 'automation', 'automating': 'automation',
        
        # Create variants
        'create': 'creation', 'created': 'creation', 'creating': 'creation',
        
        # Build variants
        'build': 'building', 'built': 'building', 'building': 'building',
        
        # Collaborate variants
        'collaborate': 'collaboration', 'collaborated': 'collaboration', 'collaborating': 'collaboration',
        
        # Communicate variants
        'communicate': 'communication', 'communicated': 'communication', 'communicating': 'communication',
        
        # Present variants
        'present': 'presentation', 'presented': 'presentation', 'presenting': 'presentation',
        
        # Organize variants
        'organize': 'organization', 'organized': 'organization', 'organizing': 'organization',
    }
    
    # Return canonical form if exists, else return original
    canonical = skill_normalization.get(skill_lower)
    if canonical:
        return canonical.title()
    
    return skill_str


def deduplicate_normalized_skills(skills_list):
    """
    Remove duplicate skills after normalization.
    Converts variants to canonical form and removes duplicates.
    """
    if not skills_list:
        return []
    
    normalized_map = {}  # canonical_form -> original_display_form
    
    for skill in skills_list:
        normalized = normalize_skill_variants(skill)
        normalized_lower = normalized.lower()
        
        # Keep first occurrence in proper case
        if normalized_lower not in normalized_map:
            normalized_map[normalized_lower] = normalized
    
    # Return deduplicated skills in sorted order
    return sorted(list(normalized_map.values()))


def is_valid_skill_string(skill_str):
    """
    Filter out invalid skill strings that are noise or artifacts.
    Returns True if skill is valid, False if it should be rejected.
    """
    normalized = normalize_skill_entry(skill_str)
    if not normalized:
        return False

    cleaned = str(skill_str).strip()

    # Reject common month abbreviations
    months = {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}
    if normalized in months:
        return False

    # Reject short uppercase fragments except widely used technical tokens
    allowed_short_tokens = {'ai', 'bi', 'ci', 'ml', 'nlp', 'sql', 'aws', 'gcp', 'api', 'etl', 'elt', 'ui', 'ux'}
    if len(normalized) <= 3 and cleaned.isupper() and normalized not in allowed_short_tokens:
        return False

    return True


def sanitize_requirement_skill_list(skills):
    """
    Remove non-skill artifacts from job requirement skill lists.
    Keeps only meaningful, DB-validated skills and strips experience phrases.
    """
    if not skills:
        return []

    sanitized = []
    seen = set()
    valid_skill_map = {s.lower(): s for s in VALID_SKILLS_SET}

    for raw_skill in skills:
        normalized = normalize_skill_entry(raw_skill)
        if not normalized:
            continue
        if normalized not in valid_skill_map:
            continue
        canonical = valid_skill_map[normalized]
        if canonical not in seen:
            seen.add(canonical)
            sanitized.append(canonical)
    return sanitized


def parse_requirement_skills(requirements_text):
    if not requirements_text:
        return []
    raw_items = [item.strip() for item in str(requirements_text).split(',') if item.strip()]
    return sanitize_requirement_skill_list(raw_items)


def validate_skills_against_db(skills, skill_db):
    """
    Validate extracted skills against database.
    Only returns skills that:
    1. Pass the valid_skill_string filter
    2. Exist in the database (exact match, case-insensitive)
    Returns the EXACT skill name from database for consistency.
    """
    validated = set()
    if not skills:
        return validated
    
    for skill in skills:
        if not skill:
            continue
        
        # First filter: check if it's a valid skill string
        if not is_valid_skill_string(skill):
            continue
        
        db_skill = resolve_skill_against_db(skill, skill_db)
        if db_skill:
            validated.add(db_skill)
    
    return validated


def extract_skills_manual(text, skill_db):
    """
    Extract skills using n-gram matching against skill database.
    Much faster than per-skill regex: O(text_tokens) vs O(skill_db_size).
    """
    if not text or not skill_db:
        return []

    text_lower = text.lower()
    skill_set = skill_db if isinstance(skill_db, set) else set(skill_db)

    # Tokenize: split on whitespace and common punctuation, keep special chars like C++, C#
    tokens = re.split(r'[\s,;:|()\[\]{}\'"]+', text_lower)
    tokens = [t.strip('.').strip() for t in tokens if t.strip() and len(t.strip()) >= 1]

    found = set()
    max_ngram = 5  # covers skills up to 5 words long

    for n in range(1, max_ngram + 1):
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            if ngram in skill_set and is_valid_skill_string(ngram):
                found.add(ngram)

    return list(found)

def extract_education_requirements(text):
    """
    Extract education requirements and minimum experience from job description.
    Returns: {"degree": "Bachelor/Master/Diploma", "majors": [], "min_gpa": 0.0, "min_experience_years": None}
    """
    if not text: return {"degree": None, "majors": [], "min_gpa": None, "min_experience_years": None}
    
    text_lower = text.lower()
    result = {"degree": None, "majors": [], "min_gpa": None, "min_experience_years": None}
    
    # Extract degree level
    degree_patterns = [
        (r"master|s2|s\.2|pascasarjana", "Master"),
        (r"bachelor|s1|s\.1|sarjana|diploma|d3|d4", "Bachelor"),
        (r"diploma|d3", "Diploma"),
    ]
    for pattern, degree in degree_patterns:
        if re.search(pattern, text_lower):
            result["degree"] = degree
            break
    
    # Extract minimum GPA
    gpa_match = re.search(r"gpa|ipk[:\s]*(\d[.,]\d{1,2})", text_lower)
    if gpa_match:
        try:
            gpa_str = gpa_match.group(1).replace(",", ".")
            result["min_gpa"] = float(gpa_str)
        except: pass
    
    # Extract minimum experience years
    exp_patterns = [
        r"(?:minimum\s+)?(?:pengalaman|experience)\s*[:]?\s*(\d+)\s*(?:\+|tahun|years?|yrs)",
        r"(\d+)\s*(?:\+|tahun|years?|yrs)\s*(?:pengalaman|experience)",
        r"(?:exp|experience)\s*[:]?\s*(\d+)\s*(?:years?|tahun)",
        r"fresh\s*graduate",  # Special case for fresh graduate
    ]
    
    for pattern in exp_patterns:
        if "fresh" in pattern and "fresh graduate" in text_lower:
            result["min_experience_years"] = 0
            break
        exp_match = re.search(pattern, text_lower)
        if exp_match:
            try:
                exp_str = exp_match.group(1)
                result["min_experience_years"] = float(exp_str)
                break
            except:
                pass
    
    # Extract majors/bidang studi - more Indonesian-specific
    major_keywords = {
        "teknik": "Teknik",
        "informatika": "Informatika",
        "computer science": "Informatika",
        "bisnis": "Bisnis",
        "manajemen": "Manajemen",
        "ekonomi": "Ekonomi",
        "akuntansi": "Akuntansi",
        "keuangan": "Keuangan",
        "marketing": "Marketing",
        "komunikasi": "Komunikasi",
        "hukum": "Hukum",
        "kedokteran": "Kedokteran",
        "seni": "Seni",
        "sastra": "Sastra",
        "desain": "Desain",
        "engineering": "Teknik",
        "statistics": "Statistika",
        "statistika": "Statistika",
        "matematika": "Matematika",
        "data science": "Data Science",
        "it": "IT",
        "industri": "Teknik Industri",
        "logistik": "Logistik",
        "supply chain": "Supply Chain",
    }
    
    for keyword, major in major_keywords.items():
        if re.search(r"\b" + keyword + r"\b", text_lower):
            if major not in result["majors"]:
                result["majors"].append(major)
    
    return result

def _normalize_degree(raw):
    """Map raw degree strings (D3, S1, Bachelor, etc.) to hierarchy key."""
    if not raw:
        return None
    d = raw.lower().strip()
    if d in ('s3', 'phd', 'ph.d', 'doktor', 'doctorate', 'doctor'):
        return 'phd'
    if d in ('s2', 'magister', 'master', "master's", 'm.sc', 'm.s', 'm.t', 'm.kom'):
        return 'master'
    if d in ('s1', 'd4', 'sarjana', 'bachelor', "bachelor's", 'bachelors'):
        return 'bachelor'
    if d in ('d3', 'd2', 'd1', 'diploma', 'ahli madya', 'a.md', 'amd'):
        return 'diploma'
    return d  # unknown — returned as-is for string comparison


def _parse_majors(raw):
    """Parse required_majors field which may be JSON array or comma string."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(m).strip() for m in raw if str(m).strip()]
    s = str(raw).strip()
    if not s:
        return []
    # Try JSON array first
    if s.startswith('['):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(m).strip() for m in parsed if str(m).strip()]
        except (ValueError, TypeError):
            pass
    # Fallback: comma-separated
    return [m.strip() for m in s.split(',') if m.strip()]


def calculate_compatibility_score(user_skills, job, user_degree=None, user_gpa=None, user_major=None, user_experience_years=None):
    score_components = {}

    DEGREE_HIERARCHY = {"diploma": 1, "bachelor": 2, "master": 3, "phd": 4, "doctorate": 4}

    # 1. SKILL MATCH SCORE (64%)
    req_list = parse_requirement_skills(job.get('requirements', ""))
    user_set = set(s.lower() for s in user_skills)
    job_set = set(r.lower() for r in req_list)

    matches = user_set.intersection(job_set)
    match_count = len(matches)
    total_reqs = len(job_set)
    skill_score = (match_count / total_reqs) * 100 if total_reqs > 0 else 0
    score_components['skill_score'] = skill_score

    # ----------------------------------------------------------------
    # 2. EDUCATION MATCH SCORE (11%)
    # Priority: use stored required_degree/required_majors from DB;
    # fall back to regex extraction if those fields are absent.
    # ----------------------------------------------------------------
    edu_req_text = extract_education_requirements(job.get('description_text', ""))

    # Use DB fields first, then regex fallback
    req_degree_raw = job.get('required_degree') or edu_req_text.get('degree')
    req_degree     = req_degree_raw  # keep original for display
    req_majors     = _parse_majors(job.get('required_majors')) or edu_req_text.get('majors') or []
    req_min_gpa    = job.get('min_gpa') or edu_req_text.get('min_gpa')

    education_score = 0
    education_status_list = []

    # Degree component (max 60 pts)
    if not req_degree:
        education_score += 60
        education_status_list.append("✓ Tidak ada persyaratan gelar khusus")
    else:
        user_norm = _normalize_degree(user_degree)
        req_norm  = _normalize_degree(req_degree)
        user_lvl  = DEGREE_HIERARCHY.get(user_norm, 0) if user_norm else 0
        req_lvl   = DEGREE_HIERARCHY.get(req_norm, 0)
        if user_degree:
            if req_lvl == 0:
                # DB stores unrecognised format — string match
                if user_norm == req_norm:
                    education_score += 60
                    education_status_list.append(f"✓ Gelar cocok: {user_degree}")
                else:
                    education_score += 20
                    education_status_list.append(f"~ Gelar berbeda: {user_degree} (syarat: {req_degree})")
            elif user_lvl >= req_lvl:
                education_score += 60
                education_status_list.append(f"✓ Gelar memenuhi syarat: {user_degree} (syarat: {req_degree})")
            else:
                education_score += 15
                education_status_list.append(f"✗ Gelar di bawah syarat: {user_degree} (syarat: {req_degree})")
        else:
            education_score += 25
            education_status_list.append(f"~ Gelar tidak terdeteksi (syarat: {req_degree})")

    # Major component (max 40 pts)
    if not req_majors:
        education_score += 40
        education_status_list.append("✓ Tidak ada persyaratan jurusan khusus")
    else:
        if user_major:
            um_lower = user_major.lower()
            rm_lower = [m.lower() for m in req_majors]
            if any(um_lower == rm or um_lower in rm or rm in um_lower for rm in rm_lower):
                education_score += 40
                education_status_list.append(f"✓ Jurusan cocok: {user_major}")
            else:
                education_score += 10
                education_status_list.append(f"~ Jurusan berbeda: {user_major} (syarat: {', '.join(req_majors)})")
        else:
            education_score += 20
            education_status_list.append(f"~ Jurusan tidak terdeteksi (syarat: {', '.join(req_majors)})")

    # GPA penalty (up to -20)
    if req_min_gpa is not None:
        try:
            min_gpa_f = float(req_min_gpa)
            if user_gpa:
                user_gpa_f = float(user_gpa) if isinstance(user_gpa, str) else float(user_gpa)
                if user_gpa_f >= min_gpa_f:
                    education_status_list.append(f"✓ IPK memenuhi syarat: {user_gpa} (min {min_gpa_f})")
                else:
                    education_score = max(0, education_score - 20)
                    education_status_list.append(f"✗ IPK di bawah syarat: {user_gpa} (min {min_gpa_f})")
            else:
                education_status_list.append(f"~ IPK tidak terdeteksi (min {min_gpa_f})")
        except (TypeError, ValueError):
            pass

    score_components['education_score'] = min(100, education_score)
    score_components['education_status'] = " | ".join(education_status_list) if education_status_list else "N/A"

    # 3. EXPERIENCE MATCH SCORE (40%)
    # Calculated independently from education match
    min_exp_years = job.get('min_experience_years')
    experience_score = 0
    experience_status = "N/A"

    if min_exp_years is None or min_exp_years == "":
        # Jika requirement pengalaman tidak ditentukan di job
        experience_score = 100
        experience_status = "Tidak ditentukan (dianggap terpenuhi)"
    elif user_experience_years is None:
        experience_score = 50
        experience_status = f"Not specified (Job requires {min_exp_years}+ years)"
    elif user_experience_years >= min_exp_years:
        experience_score = 100
        experience_status = f"✓ Qualified ({user_experience_years} years, requires {min_exp_years}+)."
    else:
        experience_score = (user_experience_years / min_exp_years) * 100 if min_exp_years > 0 else 100
        experience_status = f"✗ Below requirement ({user_experience_years} years, requires {min_exp_years}+)."

    score_components['experience_score'] = experience_score
    score_components['experience_status'] = experience_status

    # Calculate weighted total score
    total_score = (
        (skill_score * 0.64) +
        (experience_score * 0.25) +
        (score_components['education_score'] * 0.11)
    )

    return round(total_score, 1), score_components

def match_jobs(user_skills, jobs, user_degree=None, user_gpa=None, user_major=None, user_experience_years=None, user_roles=None):
    if not user_skills or not jobs: return []
    user_set = set(s.lower() for s in user_skills)
    user_roles_set = set(r.lower() for r in (user_roles or [])) if user_roles else set()
    
    # Create mapping of lowercase skill -> exact user skill name for consistent output
    user_skill_map = {s.lower(): s for s in user_skills}
    
    ranked = []
    
    for job in jobs:
        req_list = parse_requirement_skills(job.get('requirements', ""))
        if not req_list:
            continue
        job_set = set(r.lower() for r in req_list if r.strip())
        
        matches = user_set.intersection(job_set)
        match_count = len(matches)
        total_reqs = len(job_set)
        
        if total_reqs > 0:
            skill_score = (match_count / total_reqs) * 100
            # Calculate role-based match bonus
            job_title_lower = job.get('job_title', '').lower()
            job_description_lower = job.get('description_text', '').lower()
            role_match_bonus = 0
            role_match_details = []
            if user_roles_set:
                for role in user_roles_set:
                    if role in job_title_lower or job_title_lower in role:
                        role_match_bonus = 15
                        role_match_details.append(f"Title matches CV role: {role}")
                        break
                    if role in job_description_lower:
                        role_match_bonus = max(role_match_bonus, 10)
                        role_match_details.append(f"Description matches CV role: {role}")
            
            if match_count > 0 or role_match_bonus > 0:
                edu_req = extract_education_requirements(job.get('description_text', ""))
                min_exp_years = job.get('min_experience_years')

                compatibility_score, score_components = calculate_compatibility_score(
                    user_skills, job, user_degree, user_gpa, user_major, user_experience_years
                )
                
                adjusted_compatibility = min(100, compatibility_score + role_match_bonus)
                
                experience_match = score_components.get('experience_status', 'N/A')
                
                ranked.append({
                    "title": job.get('job_title', 'Unknown'),
                    "company": job.get('company_name', 'Unknown'),
                    "match_score": round(skill_score, 1),
                    "compatibility_score": adjusted_compatibility,
                    "role_match_bonus": role_match_bonus,
                    "role_match_details": role_match_details,
                    "skill_score": round(score_components.get('skill_score', 0), 1),
                    "experience_score": round(score_components.get('experience_score', 0), 1),
                    "education_score": round(score_components.get('education_score', 0), 1),
                    "matched_skills": [user_skill_map.get(m, m.title()) for m in matches],
                    "missing_skills": [m.title() for m in (job_set - user_set)],
                    "job_url": job.get('job_url', '#'),
                    "min_experience_years": min_exp_years,
                    "user_experience_match": experience_match,
                    "education_required": {
                        "degree": job.get('required_degree') or edu_req.get('degree'),
                        "majors": _parse_majors(job.get('required_majors')) or edu_req.get('majors', []),
                        "min_gpa": job.get('min_gpa') or edu_req.get('min_gpa'),
                        "min_experience_years": edu_req.get('min_experience_years')
                    }
                })
            
    # RANKING: Urutkan berdasarkan compatibility_score (tertinggi dulu)
    ranked.sort(key=lambda x: x['compatibility_score'], reverse=True)
    
    # UPDATE: Return Top 50 agar pagination di frontend ada isinya
    return ranked[:50] 

# ==============================================================================
# 4. ROUTE
# ==============================================================================

def extract_cv_section_blocks(cv_text, start_headers, end_headers):
    """Extract one or more section blocks using line-start headers."""
    if not cv_text:
        return []

    text_lower = cv_text.lower()
    collected = []

    for header in start_headers:
        match = re.search(r'(?:^|\n)[ \t]*' + re.escape(header) + r'[ \t]*(?:\n|:|$)', text_lower, re.MULTILINE)
        if not match:
            continue
        actual_start = match.end()
        end_pos = len(cv_text)
        for eh in end_headers:
            em = re.search(r'(?:^|\n)[ \t]*' + re.escape(eh) + r'[ \t]*(?:\n|:|$)', text_lower[actual_start:], re.MULTILINE)
            if em:
                pos = actual_start + em.start()
                if pos < end_pos:
                    end_pos = pos
        snippet = cv_text[actual_start:end_pos].strip()
        if snippet:
            collected.append(snippet)

    # preserve order while removing duplicates
    seen = set()
    ordered = []
    for block in collected:
        key = block.strip().lower()
        if key and key not in seen:
            seen.add(key)
            ordered.append(block)
    return ordered


def extract_targeted_skill_context(cv_text):
    """
    Skill extraction should prioritize explicit skill-bearing sections:
    Skills, Summary/Profile, and Projects.
    This avoids inflated detections from arbitrary experience prose.
    """
    if not cv_text:
        return ""

    skill_headers = [
        "technical skills", "core competencies", "technologies",
        "skills", "keahlian", "kemampuan", "expertise", "tools"
    ]
    summary_headers = [
        "summary", "professional summary", "profile", "profil",
        "ringkasan", "about me", "objective"
    ]
    project_headers = [
        "project", "projects", "proyek", "portfolio", "portofolio",
        "selected projects", "personal projects"
    ]
    end_headers = [
        "experience", "pengalaman", "education", "pendidikan",
        "certifications", "sertifikasi", "language", "bahasa",
        "work history", "achievement", "penghargaan", "interest", "hobi",
        "reference", "referensi", "honors", "awards", "publication", "publikasi"
    ]

    blocks = []
    blocks.extend(extract_cv_section_blocks(cv_text, summary_headers, end_headers))
    blocks.extend(extract_cv_section_blocks(cv_text, project_headers, end_headers))
    blocks.extend(extract_cv_section_blocks(cv_text, skill_headers, end_headers))

    if blocks:
        return "\n".join(blocks)

    # Fallback if CV headers are weak/absent: only use the top part of CV,
    # not the full text, to reduce accidental matches.
    return cv_text[:2500]


def extract_education_regex(cv_text):
    """Regex fallback to extract degree, GPA, and major from CV text."""
    if not cv_text:
        return {"degree": None, "major": None, "gpa": None}

    text = cv_text

    # --- Degree ---
    degree = None
    degree_patterns = [
        # Doctoral
        (r'\b(S\.?\s*3|Doktor|Ph\.?\s*D\.?|Doctor(?:ate)?)\b', 'PhD'),
        # Master
        (r'\b(S\.?\s*2|Magister|M\.Sc\.?|M\.S\.?|M\.Eng\.?|M\.Kom|M\.T\.?|M\.Si|M\.M|M\.Pd|Master(?:\'?s)?)\b', 'Master'),
        # Bachelor — post-nominal Indonesian (S.Kom, S.T, S.ST, S.Tr, S.Tr.T, S.Tr.Kom etc.)
        (r'\bS\.\s*(?:ST|Kom|T|Ds|E|Psi|Sos|H|Ked|Hut|Pi|IP|Farm|Si)\b', 'Bachelor'),
        # Bachelor — Sarjana Terapan (S.Tr.) = D4, treated as Bachelor
        (r'\bS\.?\s*Tr\.?(?:\s*\.\s*\w+)?', 'Bachelor'),
        # Bachelor — English abbreviations (B.S., B.Sc., B.E., B.Eng, B.Tech, B.A., B.Com)
        (r'\bB\.\s*(?:S(?:c)?|E(?:ng)?|Tech|A|Com)\.?\b', 'Bachelor'),
        # Bachelor — explicit labels and D4
        (r'\b(S\.?\s*1|D\.?\s*4|Sarjana|Bachelor(?:\'?s)?(?:\s+(?:of|degree|in))?|Undergraduate)\b', 'Bachelor'),
        # Diploma — D1/D2/D3
        (r'\b(D\.?\s*3|D\.?\s*2|D\.?\s*1|A\.?Md\.?|Ahli\s+Madya|Diploma(?:\s+\d)?|Associate(?:\'?s)?)\b', 'Diploma'),
    ]
    for pattern, label in degree_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            degree = label
            break

    # --- GPA ---
    gpa = None
    gpa_match = re.search(r'(?:IPK|GPA|IP Kumulatif|Indeks Prestasi)[^0-9]*(\d+[.,]\d+)', text, re.IGNORECASE)
    if not gpa_match:
        gpa_match = re.search(r'\b([3-4]\.\d{2})\s*/\s*4(?:\.00)?\b', text)
    if gpa_match:
        try:
            gpa = float(gpa_match.group(1).replace(',', '.'))
        except Exception:
            pass

    # --- Major ---
    # Search education section first (more precise), then fall back to
    # education-like lines across the CV. We score explicit "major/jurusan"
    # mentions higher than incidental keyword hits.
    major = None
    major_map = [
        # Applied variants (more specific) before generic entries
        (r'\bsains\s+data\s+terapan\b|\bapplied\s+data\s+science\b', 'Applied Data Science'),
        (r'\bdata\s+science\b|\bdata\s+sains\b|\bsains\s+data\b|\bilmu\s+data\b', 'Data Science'),
        (r'\bteknik\s+informatika\b|\bcomputer\s+science\b|\bilmu\s+komputer\b', 'Computer Science'),
        (r'\bsistem\s+informasi\b|\binformation\s+systems?\b', 'Information Systems'),
        (r'\bteknologi\s+informasi\b|\binformation\s+technology\b', 'Information Technology'),
        (r'\bkecerdasan\s+buatan\b|\bartificial\s+intelligence\b', 'Artificial Intelligence'),
        (r'\brekayasa\s+perangkat\s+lunak\b|\bsoftware\s+engineering\b', 'Software Engineering'),
        (r'\bteknik\s+komputer\b|\bcomputer\s+engineering\b', 'Computer Engineering'),
        # \bteknik\s+elektro\b requires whitespace — will NOT match "politeknik elektronika"
        (r'\bteknik\s+elektro\b|\belectrical\s+engineering\b', 'Electrical Engineering'),
        (r'\bteknik\s+industri\b|\bindustrial\s+engineering\b', 'Industrial Engineering'),
        (r'\bteknik\s+mesin\b|\bmechanical\s+engineering\b', 'Mechanical Engineering'),
        (r'\bteknik\s+sipil\b|\bcivil\s+engineering\b', 'Civil Engineering'),
        (r'\bmanajemen\s+informatika\b|\binformatics\s+management\b', 'Informatics Management'),
        (r'\bmanajemen\s+bisnis\b|\bbusiness\s+management\b', 'Business Management'),
        (r'\bmanajemen\b|\bmanagement\b', 'Management'),
        (r'\bakuntansi\b|\baccounting\b', 'Accounting'),
        (r'\bekonomi\b|\beconomics\b', 'Economics'),
        (r'\bstatistika\b|\bstatistics\b|\bstatistik\b', 'Statistics'),
        (r'\bmatematika\b|\bmathematics\b', 'Mathematics'),
        (r'\bfisika\b|\bphysics\b', 'Physics'),
        (r'\bkomunikasi\b|\bcommunications?\b', 'Communications'),
        (r'\bpsikologi\b|\bpsychology\b', 'Psychology'),
        (r'\bhukum\b|\blaw\b', 'Law'),
        (r'\bkedokteran\b|\bmedicine\b|\bmedical\b', 'Medicine'),
        (r'\bkeperawatan\b|\bnursing\b', 'Nursing'),
        (r'\bdesain\s+komunikasi\s+visual\b|\bvisual\s+communication\b', 'Visual Communication Design'),
        (r'\bdesain\s+grafis\b|\bgraphic\s+design\b', 'Graphic Design'),
        (r'\barsitektur\b|\barchitecture\b', 'Architecture'),
    ]

    explicit_major_patterns = [
        r'(?:major|jurusan|program studi|prodi|field of study)\s*[:\-]?\s*([A-Za-z&.,/() \-]{3,120})',
        r'(?:bachelor|master|diploma|sarjana|s\.?\s*[123]|d\.?\s*[1234])[^.\n]{0,80}?\bin\s+([A-Za-z&.,/() \-]{3,120})',
    ]

    def _pick_best_major(source_text):
        if not source_text:
            return None

        candidates = []
        source_lines = [line.strip() for line in source_text.splitlines() if line.strip()]
        proximity_hint = re.compile(
            r'\b(university|universitas|college|institut|institute|politeknik|sekolah tinggi|'
            r'bachelor|master|phd|sarjana|diploma|s\.?\s*[123]|d\.?\s*[1234])\b',
            re.IGNORECASE
        )

        for idx, line in enumerate(source_lines):
            base_score = 0
            if re.search(r'\b(major|jurusan|program studi|prodi|field of study)\b', line, re.IGNORECASE):
                base_score += 80
            if proximity_hint.search(line):
                base_score += 25

            for explicit_pattern in explicit_major_patterns:
                for explicit_match in re.finditer(explicit_pattern, line, re.IGNORECASE):
                    snippet = explicit_match.group(1).strip(" -:;,.)(")
                    for pattern, label in major_map:
                        match = re.search(pattern, snippet, re.IGNORECASE)
                        if match:
                            candidates.append((base_score + 60 + len(match.group(0)), idx, label))

            for pattern, label in major_map:
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    score = base_score + len(match.group(0))
                    candidates.append((score, idx, label))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]

    edu_section = extract_education_section(text)
    major = _pick_best_major(edu_section)

    if not major:
        education_like_lines = []
        line_anchor = re.compile(
            r'\b(education|pendidikan|academic|university|universitas|college|institut|institute|'
            r'politeknik|sekolah tinggi|major|jurusan|program studi|prodi|field of study|'
            r'bachelor|master|phd|sarjana|diploma|s\.?\s*[123]|d\.?\s*[1234])\b',
            re.IGNORECASE
        )
        for raw_line in text.splitlines():
            if line_anchor.search(raw_line):
                education_like_lines.append(raw_line)
        major = _pick_best_major("\n".join(education_like_lines))

    return {"degree": degree, "major": major, "gpa": gpa}


def resolve_major_conflict(cv_text, groq_major=None, regex_major=None):
    """
    Reconcile conflicting major predictions.
    Prefer the value that is explicitly supported by the education section,
    especially when Groq returns a broader IT family label but regex finds a
    more specific program such as Data Science.
    """
    if not groq_major:
        return regex_major
    if not regex_major:
        return groq_major
    if groq_major.strip().lower() == regex_major.strip().lower():
        return groq_major

    edu_section = extract_education_section(cv_text or "")
    edu_lower = edu_section.lower()

    alias_map = {
        "Applied Data Science": [r'\bapplied\s+data\s+science\b', r'\bsains\s+data\s+terapan\b'],
        "Data Science": [r'\bdata\s+science\b', r'\bdata\s+sains\b', r'\bsains\s+data\b', r'\bilmu\s+data\b'],
        "Computer Science": [r'\bcomputer\s+science\b', r'\bteknik\s+informatika\b', r'\bilmu\s+komputer\b'],
        "Computer Science or Informatics": [r'\bcomputer\s+science\b', r'\binformatics\b', r'\bteknik\s+informatika\b', r'\bilmu\s+komputer\b'],
        "Informatics Engineering": [r'\binformatics\s+engineering\b', r'\bteknik\s+informatika\b'],
        "Information Systems": [r'\binformation\s+systems?\b', r'\bsistem\s+informasi\b'],
        "Information Technology": [r'\binformation\s+technology\b', r'\bteknologi\s+informasi\b'],
    }

    def supported_by_education(major_name):
        patterns = alias_map.get(major_name, [r'\b' + re.escape(major_name.lower()) + r'\b'])
        return any(re.search(pattern, edu_lower, re.IGNORECASE) for pattern in patterns)

    regex_supported = supported_by_education(regex_major)
    groq_supported = supported_by_education(groq_major)

    data_science_family = {"Applied Data Science", "Data Science"}
    generic_it_family = {
        "Computer Science",
        "Computer Science or Informatics",
        "Informatics Engineering",
        "Information Systems",
        "Information Technology",
    }

    if regex_major in data_science_family and groq_major in generic_it_family and regex_supported:
        print(f"  ↳ major conflict resolved in favor of regex: {regex_major} over {groq_major}")
        return regex_major

    if regex_supported and not groq_supported:
        print(f"  ↳ major conflict resolved by education-section evidence: {regex_major} over {groq_major}")
        return regex_major

    if groq_supported and not regex_supported:
        return groq_major

    # If both are plausible, prefer the more specific non-generic result.
    if regex_major not in generic_it_family and groq_major in generic_it_family:
        print(f"  ↳ major conflict resolved by specificity: {regex_major} over {groq_major}")
        return regex_major

    return groq_major


def extract_skills_section_only(cv_text):
    """
    Ekstrak hanya bagian Skills dari CV untuk dibaca oleh NER dan Manual Extraction,
    dan SECARA EKSPLISIT mengecualikan (exclude) kata judul section itu sendiri.
    """
    if not cv_text:
        return ""

    skill_headers = [
        "technical skills", "core competencies", "technologies",
        "skills", "keahlian", "kemampuan", "expertise", "tools"
    ]
    end_headers = [
        "experience", "pengalaman", "education", "pendidikan",
        "project", "proyek", "certifications", "sertifikasi",
        "language", "bahasa", "work history", "achievement",
        "penghargaan", "interest", "hobi", "reference", "referensi",
        "honors", "awards", "publication", "publikasi", "summary", "profil"
    ]

    blocks = extract_cv_section_blocks(cv_text, skill_headers, end_headers)
    return "\n".join(blocks)

@app.route('/extract-cv/', methods=['POST', 'OPTIONS'])
def extract_cv():
    if request.method == 'OPTIONS':
        return '', 204
    
    if 'pdf' not in request.files:
        print("❌ ERROR: No file provided")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['pdf']
    
    if not file or file.filename == '':
        print("❌ ERROR: File is empty")
        return jsonify({"error": "File is empty"}), 400
    
    print(f"\n📄 PROSES FILE: {file.filename}")

    try:
        full_text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted: full_text += extracted + "\n"
        
        if not full_text.strip():
            print("❌ ERROR: No text extracted from PDF")
            return jsonify({"error": "Tidak ada teks yang dapat diekstrak dari PDF"}), 400
        
        cleaned_text = clean_text(full_text)
        name = extract_name_heuristic(full_text)
        work_exp_section = extract_work_experience_section(full_text)
        experience_years = extract_experience_years(work_exp_section) if work_exp_section else None
        
        skills_only_text = extract_skills_section_only(full_text)
        cleaned_skills_text = clean_text(skills_only_text)
        print(f"\n🎯 Skills Section Extracted ({len(cleaned_skills_text)} chars)")

        targeted_skill_context = extract_targeted_skill_context(full_text)
        cleaned_targeted_skill_context = clean_text(targeted_skill_context)
        print(f"🎯 Summary+Projects+Skills Context Extracted ({len(cleaned_targeted_skill_context)} chars)")

        # EDUCATION: regex baseline first, then Groq to improve
        print("\n🤖 Analyzing education...")
        regex_edu = extract_education_regex(full_text)
        groq_education = analyze_education_with_groq(full_text)
        # Prefer Groq for structure, but resolve major conflicts with regex
        # when the education section explicitly supports a different result.
        degree = groq_education.get('degree') or regex_edu.get('degree')
        major  = resolve_major_conflict(full_text, groq_education.get('major'), regex_edu.get('major'))
        gpa    = groq_education.get('gpa')    or regex_edu.get('gpa')

        # Hard last-resort: if degree is still null, run a broad scan over the entire text
        if not degree:
            broad_degree_patterns = [
                (r'\bS\.?\s*Tr\.?(?:\s*\.\s*\w+)?', 'Bachelor'),        # Sarjana Terapan (D4)
                (r'\bB\.\s*(?:S(?:c)?|E(?:ng)?|Tech|A|Com)\.?\b', 'Bachelor'),
                (r'\b(?:Bachelor|Sarjana|S\.?\s*1|D\.?\s*4|Undergraduate)\b', 'Bachelor'),
                (r'\bS\.\s*(?:ST|Kom|T|Ds|E|Psi|Sos|H|Farm|Si)\b', 'Bachelor'),
                (r'\b(?:Master|Magister|S\.?\s*2|M\.Sc|M\.S\.?|M\.Eng)\b', 'Master'),
                (r'\b(?:Ph\.?\s*D\.?|Doktor|S\.?\s*3)\b', 'PhD'),
                (r'\b(?:D\.?\s*[123]|A\.?Md\.?|Ahli\s+Madya|Associate)\b', 'Diploma'),
                # Institution names as last-resort degree hint
                (r'\b(?:Universitas|University|Institut\b|Institute|College|Sekolah\s+Tinggi|Politeknik)\b', 'Bachelor'),
            ]
            for pattern, label in broad_degree_patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    degree = label
                    print(f"  ↳ degree filled by broad scan: {degree}")
                    break

        # Validate major against program_studi DB — corrects & canonicalises the name
        if major:
            print(f"\n🔍 Validating major '{major}' against program_studi DB...")
            major = validate_major_against_db(major, VALID_PROGRAMS_SET)

        print(f"✓ Education: degree={degree}, major={major}, gpa={gpa}")

        # GROQ Analysis for work experiences
        print("\n🤖 Analyzing work experiences with Groq...")
        groq_analysis = analyze_cv_with_groq(full_text)

        # Use Groq data if available and better than regex
        groq_years        = groq_analysis.get('total_years')
        groq_roles        = groq_analysis.get('roles') or []
        work_experiences  = groq_analysis.get('work_experiences') or []
        groq_role_groups  = groq_analysis.get('role_groups') or []
        groq_error        = groq_analysis.get('error')
        
        if groq_years is not None:
            experience_years = groq_years
            print(f"✓ Using Groq analysis: {experience_years} years")
        elif experience_years:
            print(f"✓ Using regex extraction: {experience_years} years")
        else:
            # Fallback: accumulate date ranges from work experience section only
            date_range_years = extract_experience_from_date_ranges(work_exp_section) if work_exp_section else None
            if date_range_years:
                experience_years = date_range_years
                print(f"✓ Using date-range accumulation: {experience_years} years")
            else:
                print(f"✓ No work experience found")

        final_skills = set()
        raw_extracted_skills = []

        # ✅ NER-BASED SKILL EXTRACTION (with strict validation)
        if ner_pipeline:
            try:
                ner_input_text = cleaned_skills_text or cleaned_targeted_skill_context
                ner_results = ner_pipeline(ner_input_text[:3500])
                for r in ner_results:
                    if r['entity_group'] == 'SKILL':
                        w = re.sub(r'[^\w\+\#\.\-]', '', r['word']).strip()
                        # Strict pre-filter: reject if too short, invalid, or looks like artifact
                        if w and len(w) >= 2 and is_valid_skill_string(w):
                            raw_extracted_skills.append(w)
            except Exception as e:
                print(f"⚠️  NER Pipeline error: {e}")

        # ✅ VALIDATE AGAINST DATABASE (only skills that exist in DB)
        validated_ner_skills = validate_skills_against_db(raw_extracted_skills, VALID_SKILLS_SET)
        
        # ✅ STRUCTURED HARD-SKILL EXTRACTION from explicit skill lists
        structured_skill_candidates = extract_structured_skill_candidates(skills_only_text)
        structured_skills = validate_skills_against_db(structured_skill_candidates, VALID_SKILLS_SET)

        # ✅ MANUAL MATCHING (regex-based, only from DB skills)
        manual_skills = extract_skills_manual(
            cleaned_targeted_skill_context or cleaned_skills_text,
            VALID_SKILLS_SET
        )

        # Combine both approaches and remove duplicates (using database skill names)
        final_skills.update(validated_ner_skills)
        final_skills.update(structured_skills)
        final_skills.update(manual_skills)
        
        # Title-case for display; matching in match_jobs uses .lower() so casing doesn't affect logic
        final_skills_list = sorted([s.title() for s in set(final_skills)])

        print(f"✓ Skills Found (database-validated): {len(final_skills_list)}")
        if final_skills_list:
            print(f"  Samples: {', '.join(final_skills_list[:5])}")
        if validated_ner_skills:
            print(f"  NER validated skills: {', '.join(sorted(validated_ner_skills)[:5])}")
        if structured_skills:
            print(f"  Structured hard-skill matches: {', '.join(sorted(structured_skills)[:8])}")
        if manual_skills:
            print(f"  Manual validated skills: {', '.join(sorted(manual_skills)[:5])}")
        print(f"✓ Degree: {degree}, GPA: {gpa}")
        if groq_roles:
            print(f"✓ Job Roles (from Groq): {', '.join(groq_roles)}")
        if work_experiences:
            print(f"✓ Work Experiences Details:")
            for exp in work_experiences[:3]:  # Show top 3
                duration = exp.get('duration_years')
                if duration is None:
                    duration_text = '0.0'
                else:
                    try:
                        duration_text = f"{float(duration):.1f}"
                    except Exception:
                        duration_text = str(duration)
                print(f"   - {exp.get('position', 'N/A')} at {exp.get('company', 'N/A')} ({duration_text} years)")
        
        active_job_list = load_jobs_from_db(active_only=True)
        recs = match_jobs(
            final_skills_list,
            active_job_list,
            user_degree=degree,
            user_gpa=gpa,
            user_major=major,
            user_experience_years=experience_years,
            user_roles=groq_roles
        )

        # ✅ AGGREGATE ALL MISSING SKILLS & RECOMMENDED SKILLS FROM JOB RECOMMENDATIONS
        # Only include skills that are in the database for accuracy
        all_missing_skills = set()
        all_recommended_skills = set()
        user_skills_lower = {s.lower() for s in final_skills_list}
        db_skills_lower = {s.lower() for s in VALID_SKILLS_SET}
        
        if recs:
            for job in recs:
                # Collect all missing skills across all recommendations (only if in DB)
                if job.get('missing_skills'):
                    for skill in job['missing_skills']:
                        skill_lower = skill.lower()
                        # Only include if skill exists in database
                        if skill_lower in db_skills_lower:
                            all_missing_skills.add(skill_lower)
                
                # Get recommended skills (skills from jobs that user doesn't have yet)
                if job.get('matched_skills'):
                    for skill in job['matched_skills']:
                        skill_lower = skill.lower()
                        # Include if it's in DB and user doesn't have it
                        if skill_lower in db_skills_lower and skill_lower not in user_skills_lower:
                            all_recommended_skills.add(skill_lower)

        # Title-case for display consistency
        all_missing_skills_list = sorted([s.title() for s in all_missing_skills])
        all_recommended_skills_list = sorted([s.title() for s in all_recommended_skills])
        
        print(f"✓ Missing skills (from job requirements, DB-validated): {len(all_missing_skills_list)}")
        if all_missing_skills_list:
            print(f"  Samples: {', '.join(all_missing_skills_list[:5])}")
        print(f"✓ Recommended skills (from top jobs, DB-validated): {len(all_recommended_skills_list)}")
        if all_recommended_skills_list:
            print(f"  Samples: {', '.join(all_recommended_skills_list[:5])}")

        response = {
            "status": "success",
            "profile": {
                "name": name,
                "degree": degree,
                "major": major,
                "gpa": gpa,
                "experience_years": experience_years,
                "roles": groq_roles,
                "role_groups": groq_role_groups,
                "work_experiences": work_experiences[:5]
            },
            "groq_analysis": {
                "roles": groq_roles,
                "work_experiences": work_experiences,
                "role_groups": groq_role_groups,
                "total_years": groq_years,
                "error": groq_error
            },
            "data_last_updated": get_data_last_updated(),
            "skills_detected": final_skills_list,
            "skills_missing": all_missing_skills_list,
            "skills_recommended": all_recommended_skills_list,
            "job_recommendations": recs
        }
        
        print(f"\n✅ Response prepared: {len(final_skills_list)} skills, {len(recs)} job recommendations")
        return jsonify(response)

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

@app.route('/set-groq-key/', methods=['POST'])
def set_groq_key():
    """
    Set Groq API Key for CV analysis.
    Expected JSON: {"api_key": "your_groq_api_key"}
    """
    global groq_client
    
    data = request.get_json()
    if not data or 'api_key' not in data:
        return jsonify({"error": "Missing api_key in request"}), 400
    
    api_key = data.get('api_key', '').strip()
    
    if not api_key:
        return jsonify({"error": "API key cannot be empty"}), 400
    
    try:
        # Test the API key
        test_client = Groq(api_key=api_key)
        test_message = test_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=10,
            messages=[{"role": "user", "content": "Test"}]
        )
        
        # If successful, set the global client
        groq_client = test_client
        os.environ["GROQ_API_KEY"] = api_key
        
        print(f"✅ Groq API Key Set Successfully!")
        return jsonify({
            "status": "success",
            "message": "Groq API Key configured successfully",
            "groq_initialized": True
        })
    
    except Exception as e:
        print(f"❌ Invalid Groq API Key: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to validate API key: {str(e)}",
            "groq_initialized": False
        }), 400

def page_text_indicates_expired(text):
    if not text:
        return False
    text_lower = text.lower()
    expired_phrases = [
        "lowongan kerja ini tidak lagi diiklankan",
        "lowongan pekerjaan tidak lagi diiklankan",
        "lowongan ini tidak lagi diiklankan",
        "lowongan tidak lagi diiklankan",
        "tidak lagi diiklankan",
        "lowongan tidak diiklankan",
        "lowongan tidak tersedia",
        "lowongan sudah tidak tersedia",
        "lowongan sudah ditutup",
        "lowongan sudah berakhir",
        "lowongan sudah tidak aktif",
    ]

    for phrase in expired_phrases:
        if phrase in text_lower:
            return True
    return False


def check_url_status(url, timeout=5):
    """
    Check if URL is still accessible and active.
    Returns: {"url": url, "status": "active|expired|error", "status_code": int or None, "reason": str}
    """
    if not url or url == '#':
        return {"url": url, "status": "error", "status_code": None, "reason": "Invalid URL"}
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Try HEAD request first (faster)
        try:
            response = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
        except Exception as e:
            # If HEAD fails (some servers block HEAD), try GET as a fallback
            try:
                get_response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
                if 200 <= get_response.status_code < 300:
                    body = get_response.text
                    if page_text_indicates_expired(body):
                        return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": "Page indicates job is no longer advertised"}
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "OK (GET fallback)"}
                elif 300 <= get_response.status_code < 400:
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "Redirect (OK, GET fallback)"}
                else:
                    return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": f"HTTP {get_response.status_code} on GET fallback"}
            except Exception as e2:
                return {"url": url, "status": "expired", "status_code": None, "reason": f"HEAD+GET error: {str(e2)}"}

        # Status codes that usually mean NOT available; but some servers return 403/401 for HEAD
        expired_codes = [404, 410]

        status_code = response.status_code

        # Treat server errors as expired
        if status_code >= 500:
            return {"url": url, "status": "expired", "status_code": status_code, "reason": f"HTTP {status_code}"}

        # If HEAD indicates success (2xx or 3xx), perform GET to inspect page text
        if 200 <= status_code < 400:
            try:
                get_response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
                if 200 <= get_response.status_code < 300:
                    body = get_response.text
                    if page_text_indicates_expired(body):
                        return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": "Page indicates job is no longer advertised"}
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "OK"}
                elif 300 <= get_response.status_code < 400:
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "Redirect (OK)"}
                else:
                    return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": f"HTTP {get_response.status_code} on GET request"}
            except Exception as e:
                return {"url": url, "status": "expired", "status_code": None, "reason": f"GET error: {str(e)}"}

        # For common client errors on HEAD (e.g., 401,403,405,400) try GET before marking expired
        if status_code in (400, 401, 403, 405):
            try:
                get_response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
                if 200 <= get_response.status_code < 300:
                    body = get_response.text
                    if page_text_indicates_expired(body):
                        return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": "Page indicates job is no longer advertised"}
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "OK (GET confirmed)"}
                elif 300 <= get_response.status_code < 400:
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "Redirect (OK, GET confirmed)"}
                else:
                    return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": f"HTTP {get_response.status_code} on GET after HEAD {status_code}"}
            except Exception as e:
                return {"url": url, "status": "expired", "status_code": None, "reason": f"HEAD {status_code} then GET error: {str(e)}"}

        # If none of the above matched, consider it expired
        return {"url": url, "status": "expired", "status_code": status_code, "reason": f"HTTP {status_code}"}
    
    except requests.exceptions.Timeout:
        return {
            "url": url, 
            "status": "expired", 
            "status_code": None, 
            "reason": "Timeout - Server tidak merespon"
        }
    except requests.exceptions.ConnectionError:
        return {
            "url": url, 
            "status": "expired", 
            "status_code": None, 
            "reason": "Connection error - Tidak dapat terhubung"
        }
    except Exception as e:
        return {
            "url": url, 
            "status": "expired", 
            "status_code": None, 
            "reason": f"Error: {str(e)}"
        }

# ==============================================================================
# Background URL checking: state + worker
# ==============================================================================
# Status pengecekan disimpan di memori, dilindungi lock agar aman diakses
# dari thread background maupun request HTTP secara bersamaan.
CHECK_STATE = {
    "running": False,
    "done": False,
    "total": 0,
    "checked": 0,
    "active": 0,
    "expired": 0,
    "errors": 0,
    "results": [],
    "error": None,
    "started_at": None,
    "finished_at": None,
}
CHECK_LOCK = threading.Lock()


def _run_link_check():
    """Worker yang berjalan di thread background: memeriksa semua URL aktif,
    memperbarui CHECK_STATE secara berkala agar frontend bisa polling progres."""
    try:
        jobs_data = load_jobs_from_db()
        active_jobs = [job for job in jobs_data if job.get('status') == 'active']
        total = len(active_jobs)
        print(f"\n🔍 [BG] Mulai cek URL untuk {total} lowongan aktif...")

        with CHECK_LOCK:
            CHECK_STATE.update(total=total, checked=0, active=0,
                               expired=0, errors=0, results=[])

        results = []
        active_count = expired_count = error_count = 0
        expired_rowids = []

        # Lebih banyak worker = lebih cepat. Timeout per URL diperpendek ke 4 detik.
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(check_url_status, job.get('job_url', '#'), 4): idx
                       for idx, job in enumerate(active_jobs)}

            for future in as_completed(futures):
                idx = futures[future]
                job = active_jobs[idx]
                try:
                    check_result = future.result()
                    status = check_result.get('status', 'error')
                    if status == 'active':
                        job['status'] = 'active'
                        active_count += 1
                    else:  # 'expired' atau 'error' -> nonaktifkan
                        job['status'] = 'expired'
                        if update_job_active_state(job.get('id'), False):
                            expired_rowids.append(job.get('id'))
                        expired_count += 1
                    results.append({
                        "id": job.get('id'),
                        "title": job.get('job_title', 'Unknown'),
                        "url": check_result.get('url'),
                        "status": job['status'],
                        "status_code": check_result.get('status_code'),
                        "reason": check_result.get('reason', '')
                    })
                except Exception as e:
                    print(f"❌ [BG] Error cek job {idx + 1}: {e}")
                    error_count += 1
                    results.append({
                        "id": job.get('id'),
                        "title": job.get('job_title', 'Unknown'),
                        "url": job.get('job_url', '#'),
                        "status": "error",
                        "reason": str(e)
                    })

                # Perbarui progres agar bisa dipantau frontend
                with CHECK_LOCK:
                    CHECK_STATE.update(checked=len(results), active=active_count,
                                       expired=expired_count, errors=error_count)

        if expired_rowids:
            try:
                JOB_LIST[:] = load_jobs_from_db()
            except Exception as e:
                print(f"⚠️ [BG] Gagal refresh JOB_LIST: {e}")

        with CHECK_LOCK:
            CHECK_STATE.update(running=False, done=True, results=results,
                               active=active_count, expired=expired_count,
                               errors=error_count, checked=len(results),
                               finished_at=datetime.datetime.now().isoformat())
        print(f"✅ [BG] Selesai: {active_count} aktif, {expired_count} expired, {error_count} error.")

    except Exception as e:
        print(f"❌ [BG] Error fatal di _run_link_check: {e}")
        with CHECK_LOCK:
            CHECK_STATE.update(running=False, done=True, error=str(e),
                               finished_at=datetime.datetime.now().isoformat())


@app.route('/api/check-links', methods=['POST', 'OPTIONS'])
def check_links():
    """Memulai pengecekan link di latar belakang lalu LANGSUNG membalas,
    sehingga browser tidak timeout. Progres & hasil diambil lewat
    GET /api/check-links/status."""
    if request.method == 'OPTIONS':
        return '', 204

    with CHECK_LOCK:
        if CHECK_STATE.get("running"):
            return jsonify({
                "status": "already_running",
                "message": "Pengecekan sedang berjalan.",
                "total": CHECK_STATE.get("total", 0),
                "checked": CHECK_STATE.get("checked", 0)
            }), 200
        CHECK_STATE.update(running=True, done=False, total=0, checked=0,
                           active=0, expired=0, errors=0, results=[],
                           error=None,
                           started_at=datetime.datetime.now().isoformat(),
                           finished_at=None)

    thread = threading.Thread(target=_run_link_check, daemon=True)
    thread.start()

    return jsonify({
        "status": "started",
        "message": "Pengecekan link dimulai di latar belakang."
    }), 202


@app.route('/api/check-links/status', methods=['GET', 'OPTIONS'])
def check_links_status():
    """Mengembalikan progres/hasil pengecekan terakhir. Saat selesai (done=True),
    menyertakan ringkasan dan array results lengkap agar kompatibel dengan dialog
    hasil di frontend."""
    if request.method == 'OPTIONS':
        return '', 204

    with CHECK_LOCK:
        snapshot = dict(CHECK_STATE)

    payload = {
        "status": "success",
        "running": snapshot.get("running", False),
        "done": snapshot.get("done", False),
        "total": snapshot.get("total", 0),
        "checked": snapshot.get("checked", 0),
        "active": snapshot.get("active", 0),
        "expired": snapshot.get("expired", 0),
        "errors": snapshot.get("errors", 0),
    }
    if snapshot.get("done"):
        payload["results"] = snapshot.get("results", [])
        if snapshot.get("error"):
            payload["error"] = snapshot["error"]
    return jsonify(payload), 200

@app.route('/api/delete-expired', methods=['POST', 'OPTIONS'])
def delete_expired():
    """
    Mark all expired jobs as deactivated in database.
    Returns:
    {
        "status": "success",
        "deactivated_count": int,
        "expired_count": int,
        "active_count": int,
        "total_count": int,
        "deactivated_jobs": [...]
    }
    """
    if request.method == 'OPTIONS':
        return '', 204

    global JOB_LIST
    try:
        print("\n🛑 Marking expired jobs as deactivated...")
        all_jobs = load_jobs_from_db(active_only=False)
        expired = [j for j in all_jobs if j.get('status') == 'expired']

        deactivated_jobs = []
        for job in expired:
            if update_job_active_state(job.get('id'), False):
                deactivated_jobs.append({
                    "id": job.get('id'),
                    "title": job.get('job_title', 'Unknown'),
                    "company": job.get('company_name', 'Unknown'),
                    "url": job.get('job_url', '#')
                })

        deactivated_count = len(deactivated_jobs)

        JOB_LIST = load_jobs_from_db(active_only=True)
        active_count = len(JOB_LIST)
        expired_count = len(load_jobs_from_db(active_only=False)) - active_count
        total_count = active_count + expired_count

        print(f"✅ Marked {deactivated_count} expired jobs as deactivated ({active_count} active, {expired_count} expired)")

        return jsonify({
            "status": "success",
            "deactivated_count": deactivated_count,
            "expired_count": expired_count,
            "active_count": active_count,
            "total_count": total_count,
            "deactivated_jobs": deactivated_jobs
        })

    except Exception as e:
        print(f"❌ Error in delete_expired: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route('/api/skills', methods=['GET', 'POST', 'DELETE', 'OPTIONS'])
def manage_skills():
    """Manage skills database"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if request.method == 'GET':
            rows = fetch_db_rows("SELECT rowid AS id, skill AS name FROM skills ORDER BY skill")
            filtered_rows = []
            for row in rows:
                normalized = normalize_skill_entry(row.get('name'))
                if normalized and is_relevant_technical_skill(normalized):
                    filtered_rows.append({
                        "id": row.get('id'),
                        "name": row.get('name')
                    })
            print(f"✅ Returned {len(filtered_rows)} filtered skills from DB")
            return jsonify(filtered_rows)

        elif request.method == 'POST':
            data = request.get_json()
            skill_name = (data.get('name') or '').strip()
            if not skill_name:
                return jsonify({"error": "Skill name required"}), 400
            skill_lower = normalize_skill_entry(skill_name)
            if not skill_lower:
                return jsonify({"error": "Skill tidak valid atau tidak bermakna untuk database skill"}), 400
            if not is_relevant_technical_skill(skill_lower):
                return jsonify({"error": "Skill tidak relevan untuk domain IT / Data Science"}), 400
            if skill_lower in VALID_SKILLS_SET:
                return jsonify({"error": f"Skill '{skill_name}' sudah ada"}), 409
            execute_db_change("INSERT INTO skills (skill) VALUES (?)", (skill_name,))
            VALID_SKILLS_SET.add(skill_lower)
            print(f"✅ Added skill to DB: {skill_name}")
            return jsonify({"status": "success", "message": f"Skill '{skill_name}' ditambahkan"}), 201

        elif request.method == 'DELETE':
            skill_id = request.args.get('id')
            if not skill_id:
                return jsonify({"error": "ID required"}), 400
            rows = fetch_db_rows("SELECT rowid AS id, skill AS name FROM skills WHERE rowid = ?", (int(skill_id),))
            if not rows:
                return jsonify({"error": "Skill not found"}), 404
            skill_name = rows[0]['name']
            execute_db_change("DELETE FROM skills WHERE rowid = ?", (int(skill_id),))
            VALID_SKILLS_SET.discard(skill_name.lower())
            print(f"✅ Deleted skill from DB: {skill_name}")
            return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"❌ Error in manage_skills: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs', methods=['GET', 'POST', 'DELETE', 'OPTIONS'])
def manage_jobs():
    """Manage jobs database"""
    if request.method == 'OPTIONS':
        return '', 204

    global JOB_LIST
    try:
        if request.method == 'GET':
            include_inactive = str(request.args.get('include_inactive', '')).strip().lower() in ('1', 'true', 'yes', 'all')
            status_filter = (request.args.get('status') or '').strip().lower()

            all_data   = load_jobs_from_db(active_only=False)
            active_count  = sum(1 for j in all_data if j.get('status') != 'expired')
            expired_count = len(all_data) - active_count

            jobs_list = []
            visible_jobs = all_data if include_inactive else [j for j in all_data if j.get('status') != 'expired']
            if status_filter in ('active', 'expired'):
                visible_jobs = [j for j in visible_jobs if j.get('status') == status_filter]

            for idx, job in enumerate(visible_jobs, 1):
                jobs_list.append({
                    "id":       job.get('id'),
                    "list_id":  idx,
                    "title":    job.get('job_title', 'Unknown'),
                    "company":  job.get('company_name', 'Unknown'),
                    "url":      job.get('job_url', '#'),
                    "postDate": None,
                    "scrapedAt": None,
                    "status":   job.get('status', 'active')
                })

            print(f"✅ Returned {len(jobs_list)} jobs (include_inactive={include_inactive}, active={active_count}, expired={expired_count})")
            return jsonify({
                "jobs":          jobs_list,
                "active_count":  active_count,
                "expired_count": expired_count,
                "total_count":   len(all_data),
                "data_last_updated": get_data_last_updated()
            })
        
        elif request.method == 'POST':
            # Upload pekerjaan: terima file CSV/Excel, proses, simpan ke DB.
            try:
                print("📥 [UPLOAD] Request POST /api/jobs diterima.")
                if 'file' not in request.files:
                    print("❌ [UPLOAD] Field 'file' tidak ada di request.")
                    return jsonify({"error": "Tidak ada file yang diunggah (field 'file' kosong)."}), 400

                uploaded = request.files['file']
                if not uploaded or uploaded.filename == '':
                    print("❌ [UPLOAD] Nama file kosong.")
                    return jsonify({"error": "Nama file kosong."}), 400

                print(f"📄 [UPLOAD] Memproses file: {uploaded.filename}")
                import time as _time; _t0 = _time.time()
                try:
                    processed_jobs, err = process_uploaded_jobs_file(uploaded)
                except Exception as e:
                    print(f"❌ [UPLOAD] Error processing uploaded file: {e}")
                    import traceback; traceback.print_exc()
                    return jsonify({"error": f"Gagal memproses file: {e}"}), 400

                if err:
                    return jsonify({"error": err}), 400
                if not processed_jobs:
                    return jsonify({"error": "Tidak ada baris pekerjaan yang valid di file."}), 400

                parse_elapsed = _time.time() - _t0
                print(f"⏱  [UPLOAD] Parsing file selesai dalam {parse_elapsed:.2f}s")
                print(f"📥 [UPLOAD] Memasukkan {len(processed_jobs)} pekerjaan ke database...")

                inserted = 0
                failed = 0
                inserted_jobs = []  # (rowid, description_text)
                for job in processed_jobs:
                    rid = insert_job_to_db(job)
                    if rid is not None:
                        inserted += 1
                        inserted_jobs.append((rid, job.get('description_text', '')))
                    else:
                        failed += 1

                db_elapsed = _time.time() - _t0
                print(f"✅ [UPLOAD] Inserted ke DB: {inserted} sukses, {failed} gagal — total {db_elapsed:.2f}s")

                try:
                    JOB_LIST = load_jobs_from_db()
                    print(f"✅ [UPLOAD] JOB_LIST diperbarui: {len(JOB_LIST)} lowongan aktif")
                except Exception as e:
                    print(f"⚠️ Gagal refresh JOB_LIST (data tetap tersimpan): {e}")

                # Skill extraction dijalankan di background agar response langsung dikirim
                if inserted_jobs:
                    def _bg_extract(jobs_snapshot, t_start):
                        import time as _time
                        print(f"\n🔧 [BG-SKILL] Mulai ekstraksi skill untuk {len(jobs_snapshot)} lowongan...")
                        done = 0
                        for rid, text in jobs_snapshot:
                            if not text:
                                continue
                            try:
                                skills = extract_skills_manual(text.lower(), VALID_SKILLS_SET)
                                if skills:
                                    skills = sanitize_requirement_skill_list(skills)
                                if skills:
                                    execute_db_change(
                                        "UPDATE jobs SET requirements = ? WHERE rowid = ?",
                                        (', '.join(skills), rid)
                                    )
                                done += 1
                            except Exception as ex:
                                print(f"⚠️ [BG-SKILL] Error rowid={rid}: {ex}")
                        elapsed = _time.time() - t_start
                        print(f"✅ [BG-SKILL] Selesai: {done}/{len(jobs_snapshot)} lowongan diperbarui ({elapsed:.1f}s)\n")
                    threading.Thread(target=_bg_extract, args=(inserted_jobs, _t0), daemon=True).start()
                    print(f"🔧 [UPLOAD] Background skill extraction dimulai ({len(inserted_jobs)} job) — berlangsung di latar belakang")

                total_elapsed = _time.time() - _t0
                print(f"✅ [UPLOAD] Upload selesai dalam {total_elapsed:.2f}s: {inserted} ditambahkan, {failed} gagal.")
                print(f"{'='*55}\n")
                return jsonify({
                    "status": "success",
                    "inserted": inserted,
                    "failed": failed,
                    "total_processed": len(processed_jobs),
                    "message": f"{inserted} pekerjaan berhasil ditambahkan ke database."
                }), 200
            except Exception as e:
                print(f"❌ [UPLOAD] Unexpected error during POST /api/jobs: {e}")
                import traceback; traceback.print_exc()
                return jsonify({"error": f"Upload gagal: {str(e)}"}), 500

        elif request.method == 'DELETE':
            job_id = request.args.get('id')
            if job_id:
                try:
                    rowid = resolve_job_rowid(job_id)
                    if rowid is not None and delete_job_from_db(rowid):
                        JOB_LIST = load_jobs_from_db(active_only=True)
                        print(f"✅ Deleted job rowid: {rowid}")
                        return jsonify({"status": "success"}), 200
                except Exception as e:
                    print(f"❌ Error deleting job by ID: {e}")
            return jsonify({"error": "Job not found"}), 404
    
    except Exception as e:
        print(f"❌ Unexpected error in manage_jobs ({request.method}): {str(e)}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    print("🚀 Server berjalan di PORT 5002...")
    print("\n📡 Available Endpoints:")
    print("   POST   /extract-cv/          - Extract CV and analyze skills")
    print("   POST   /set-groq-key/        - Set Groq API Key")
    print("   GET    /api/skills           - Get all skills")
    print("   POST   /api/skills           - Add new skill")
    print("   DELETE /api/skills?id=X      - Delete skill by ID")
    print("   GET    /api/jobs             - Get all jobs")
    print("   DELETE /api/jobs?id=X        - Delete job by ID")    
    print("   POST   /api/check-links      - Check status of all job URLs")
    print("   POST   /api/delete-expired   - Delete all expired jobs")
    # On Windows the automatic reloader can trigger select() on non-socket handles
    # which leads to OSError: [WinError 10038]. Disable the reloader to avoid this.
    app.run(host='127.0.0.1', debug=False, port=5002, use_reloader=False, threaded=True)
