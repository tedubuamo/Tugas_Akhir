import os
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
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from groq import Groq

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
    return response




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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


def load_skills_from_db():
    rows = fetch_db_rows("SELECT skill FROM skills WHERE skill IS NOT NULL")
    cleaned = set()
    for row in rows:
        if row.get('skill'):
            skill = row['skill'].strip().rstrip(';').strip().lower()
            if skill and len(skill) >= 2:
                cleaned.add(skill)
    return cleaned


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
        return [], "File kosong atau tidak terbaca."

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
        plain = clean_html_to_text(raw_html) if raw_html else (str(row.get(desc_col)) if desc_col and not pd.isna(row.get(desc_col)) else "")
        text_for_extraction = plain.lower()

        # Skill extraction reuses the existing manual matcher + skill DB
        skills = extract_skills_manual(text_for_extraction, VALID_SKILLS_SET)

        # Education / experience extraction reuses existing function
        edu = extract_education_requirements(text_for_extraction)

        def safe(colname):
            if not colname:
                return None
            v = row.get(colname)
            if pd.isna(v):
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
            # requirements stored as a comma-joined string (matches existing schema)
            "requirements": ", ".join(skills) if skills else "",
            "min_experience_years": edu.get("min_experience_years"),
            "min_gpa": edu.get("min_gpa"),
            "required_degree": edu.get("degree"),
            "required_majors": ", ".join(edu.get("majors", [])) if edu.get("majors") else "",
        })

    return processed, None


VALID_SKILLS_SET = load_skills_from_db()
print(f"✅ Skills Loaded from SQLite: {len(VALID_SKILLS_SET)}")

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
    Extract only the Work Experience section from CV text.
    Returns: extracted work experience section or empty string if not found
    """
    if not cv_text:
        return ""
    
    text_lower = cv_text.lower()
    
    # Keywords untuk mendeteksi awal section work experience
    start_keywords = [
        "work experience",
        "professional experience",
        "experience",
        "employment history",
        "job history",
        "pengalaman kerja",
        "pengalaman",
        "riwayat pekerjaan",
        "pekerjaan"
    ]
    
    # Keywords untuk mendeteksi akhir section (section berikutnya)
    end_keywords = [
        "education",
        "pendidikan",
        "skills",
        "keahlian",
        "skill",
        "certification",
        "sertifikasi",
        "award",
        "penghargaan",
        "project",
        "proyek",
        "portfolio",
        "portofolio",
        "language",
        "bahasa",
        "reference",
        "referensi"
    ]
    
    start_pos = -1
    end_pos = len(cv_text)
    
    # Find start position
    for keyword in start_keywords:
        pos = text_lower.find(keyword)
        if pos != -1:
            start_pos = pos
            break
    
    if start_pos == -1:
        # Work experience section not found
        return ""
    
    # Find end position (next section)
    for keyword in end_keywords:
        # Search after work experience section
        pos = text_lower.find(keyword, start_pos + 20)
        if pos != -1 and pos < end_pos:
            end_pos = pos
    
    work_exp_section = cv_text[start_pos:end_pos].strip()
    return work_exp_section

def extract_education_section(cv_text):
    """
    Extract only the Education section from CV text to save Groq tokens.
    """
    if not cv_text:
        return ""
    
    text_lower = cv_text.lower()
    start_keywords = ["education", "pendidikan", "academic background", "riwayat pendidikan"]
    end_keywords = ["experience", "pengalaman", "skills", "keahlian", "project", "proyek", "certification", "sertifikasi", "language", "bahasa", "work history"]
    
    start_pos = -1
    end_pos = len(cv_text)
    
    # Find start position
    for keyword in start_keywords:
        pos = text_lower.find(keyword)
        if pos != -1:
            start_pos = pos
            break
            
    if start_pos == -1:
        # Jika tidak ada header pendidikan yang jelas, ambil 2000 karakter pertama saja
        return cv_text[:2000]
        
    # Find end position (next section)
    for keyword in end_keywords:
        pos = text_lower.find(keyword, start_pos + 15)
        if pos != -1 and pos < end_pos:
            end_pos = pos
            
    return cv_text[start_pos:end_pos].strip()

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
        
        prompt = f"""Analyze the following Work Experience section from CV and extract detailed information in JSON format.

                Work Experience Section:
                {work_exp_section[:3500]}

                Please extract and return a JSON object with:
                1. "work_experiences": array of objects with {{
                "company": company name,
                "position": job position/role,
                "start_year": start year (number),
                "end_year": end year (number or "present"),
                "duration_years": calculated duration in years (float)
                }}
                2. "total_years": total years of experience (float) - sum of all durations
                3. "roles": array of all job roles/positions extracted

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
        return {"degree": None, "major": None, "gpa": None, "error": "Groq not available"}
        
    try:
        edu_section = extract_education_section(cv_text)
        print(f"🎓 Education Section Extracted ({len(edu_section)} chars)")
        
        prompt = f"""Analyze the following Education section from an Indonesian CV and extract the details in JSON format.
        
        Education Section:
        {edu_section[:2500]}
        
        Extract exactly these 3 fields:
        1. "degree": The highest degree level. Map the Indonesian abbreviations strictly to ONE of these English standards:
           - "Doctorate" (for S3, Doktor, Dr.)
           - "Master" (for S2, Magister, M.Sc, M.Kom, M.T, dll)
           - "Bachelor" (for S1, D4, Sarjana, S.ST, S.Kom, S.T, S.Ds, S.E, dll)
           - "Diploma" (for D3, D2, D1, A.Md, Ahli Madya, dll)
           If none found, return null.
        2. "major": The major or field of study TRANSLATED TO ENGLISH in title case. 
           (e.g., if the CV says "Data Sains Terapan", output "Applied Data Science". If "Teknik Informatika", output "Informatics Engineering" or "Computer Science". If "Sistem Informasi", output "Information Systems"). 
           If none found, return null.
        3. "gpa": The GPA (IPK) as a float number (e.g., 3.8). If none found, return null.
        
        Return ONLY valid JSON, no markdown or extra text."""

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, 
            max_tokens=200,
            response_format={"type": "json_object"}, 
            top_p=1,
        )
        
        response_text = completion.choices[0].message.content.strip()
        result = json.loads(response_text)
        
        print(f"🔍 GROQ Education Result: Degree: {result.get('degree')}, Major: {result.get('major')}, GPA: {result.get('gpa')}")
        return result
        
    except Exception as e:
        print(f"❌ Groq Education Analysis Error: {str(e)}")
        return {"degree": None, "major": None, "gpa": None, "error": str(e)}

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
    if not skill_str:
        return False
    
    cleaned = str(skill_str).strip()
    
    # Reject single characters or very short strings
    if len(cleaned) < 2:
        return False
    
    # Reject all-caps single letters (like "S", "R", "In", "Pt")
    if len(cleaned) <= 2 and cleaned.isupper():
        return False
    
    # Reject common month abbreviations
    months = {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}
    if cleaned.lower() in months:
        return False
    
    # Reject prepositions and small words that are artifacts
    artifacts = {
        'to', 'in', 'on', 'at', 'by', 'of', 'or', 'and', 'the', 'a', 'is', 'are',
        'be', 'was', 'were', 'com', 'pt', 'ul', 'ui', 'as', 'it', 'if', 'so',
        'up', 'no', 'go', 'do', 'me', 'my', 'we', 'he', 'im', 'id'
    }
    if cleaned.lower() in artifacts:
        return False
    
    # Reject single words without technical meaning (too generic)
    generic_words = {
        'data', 'web', 'work', 'east', 'time', 'help',
        'user', 'system', 'process', 'using'
    }
    if cleaned.lower() in generic_words:
        return False
    
    # Reject if it's mostly special characters
    alpha_count = sum(1 for c in cleaned if c.isalnum())
    if alpha_count < len(cleaned) * 0.7:  # Less than 70% alphanumeric
        return False
    
    return True


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
    
    # Create a mapping of lowercase skill -> database skill for exact retrieval
    db_skill_map = {s.lower(): s for s in skill_db}
    
    for skill in skills:
        if not skill:
            continue
        
        # First filter: check if it's a valid skill string
        if not is_valid_skill_string(skill):
            continue
        
        normalized = skill.strip().lower()
        
        # Second filter: must exist in database (case-insensitive match)
        if normalized in db_skill_map:
            # Use the EXACT skill name from database for perfect consistency
            db_skill = db_skill_map[normalized]
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
    skill_set = set(skill_db)  # O(1) lookup

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

def calculate_compatibility_score(user_skills, job, user_degree=None, user_gpa=None, user_major=None, user_experience_years=None):
    score_components = {}

    # 1. SKILL MATCH SCORE (40%)
    reqs = job.get('requirements', "")
    req_list = [r.strip() for r in reqs.split(',') if r.strip()]
    user_set = set(s.lower() for s in user_skills)
    job_set = set(r.lower() for r in req_list)

    matches = user_set.intersection(job_set)
    match_count = len(matches)
    total_reqs = len(job_set)

    if total_reqs > 0:
        skill_score = (match_count / total_reqs) * 100
    else:
        skill_score = 0

    score_components['skill_score'] = skill_score

    # Extract job education requirements
    edu_req = extract_education_requirements(job.get('description_text', ""))

    # Degree / major match logic
    degree_match = True
    major_match = True
    if edu_req.get('degree'):
        if user_degree:
            degree_hierarchy = {"diploma": 1, "bachelor": 2, "master": 3}
            user_level = degree_hierarchy.get(user_degree.lower(), 0)
            req_level = degree_hierarchy.get(edu_req.get('degree', '').lower(), 0)
            degree_match = user_level >= req_level and req_level > 0
        else:
            degree_match = False

    if edu_req.get('majors'):
        if user_major:
            major_match = any(user_major.lower() == major.lower() for major in edu_req.get('majors', []))
        else:
            major_match = False

    # 2. EDUCATION MATCH SCORE (20%)
    education_score = 0
    education_status_list = []

    # Jika degree dan major tidak cocok sama sekali, education score = 0
    if not degree_match and not major_match:
        education_score = 0
        education_status_list.append("✗ Degree and major do not match")
    else:
        if not edu_req.get('degree') and not edu_req.get('majors'):
            education_score = 100
            education_status_list.append("✓ Tidak ada persyaratan pendidikan khusus")
        else:
            if edu_req.get('degree'):
                if user_degree:
                    if user_degree.lower() in edu_req.get('degree', '').lower():
                        education_score += 50
                        education_status_list.append(f"✓ Degree matches: {user_degree}")
                    else:
                        degree_hierarchy = {"diploma": 1, "bachelor": 2, "master": 3}
                        user_level = degree_hierarchy.get(user_degree.lower(), 0)
                        req_level = degree_hierarchy.get(edu_req.get('degree', '').lower(), 0)
                        if user_level >= req_level and user_level > 0 and req_level > 0:
                            education_score += 40
                            education_status_list.append(f"~ Degree higher than required: {user_degree} (requires {edu_req.get('degree')})")
                        else:
                            education_score += 10
                            education_status_list.append(f"✗ Degree mismatch: {user_degree} (requires {edu_req.get('degree')})")
                else:
                    education_score += 20
                    education_status_list.append(f"~ Degree not specified in CV (requires {edu_req.get('degree')})")
            else:
                education_score += 50
                education_status_list.append("✓ No specific degree requirement")

            if edu_req.get('majors'):
                if user_major:
                    if any(user_major.lower() == major.lower() for major in edu_req.get('majors', [])):
                        education_score += 40
                        education_status_list.append(f"✓ Major matches: {user_major}")
                    else:
                        education_score += 10
                        education_status_list.append(f"✗ Major mismatch: {user_major} (requires {', '.join(edu_req.get('majors'))})")
                else:
                    education_score += 20
                    education_status_list.append(f"~ Major not specified in CV (requires {', '.join(edu_req.get('majors'))})")
            else:
                education_score += 40
                education_status_list.append("✓ No specific major requirement")

        if edu_req.get('min_gpa') is not None:
            if user_gpa:
                user_gpa_float = float(user_gpa) if isinstance(user_gpa, str) else user_gpa
                if user_gpa_float >= edu_req.get('min_gpa'):
                    education_status_list.append(f"✓ GPA meets requirement: {user_gpa} (requires {edu_req.get('min_gpa')})")
                else:
                    education_score = max(0, education_score - 20)
                    education_status_list.append(f"✗ GPA below requirement: {user_gpa} (requires {edu_req.get('min_gpa')})")
            else:
                education_status_list.append(f"~ GPA not specified in CV (requires {edu_req.get('min_gpa')})")

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
        reqs = job.get('requirements', "")
        if not reqs: continue
        req_list = [r.strip() for r in reqs.split(',')]
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
                        "degree": edu_req.get('degree'),
                        "majors": edu_req.get('majors', []),
                        "min_gpa": edu_req.get('min_gpa'),
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
def extract_skills_section_only(cv_text):
    """
    Ekstrak hanya bagian Skills dari CV untuk dibaca oleh NER dan Manual Extraction,
    dan SECARA EKSPLISIT mengecualikan (exclude) kata judul section itu sendiri.
    """
    if not cv_text:
        return ""
        
    text_lower = cv_text.lower()
    
    # Keyword untuk mendeteksi awal section skill
    skill_headers = [
        "technical skills", "core competencies", "technologies", 
        "skills", "keahlian", "kemampuan", "expertise", "tools"
    ]
    
    # Keyword untuk mendeteksi section berikutnya (sebagai batas akhir pembacaan)
    end_headers = [
        "experience", "pengalaman", "education", "pendidikan", 
        "project", "proyek", "certifications", "sertifikasi",
        "language", "bahasa", "work history", "achievement",
        "penghargaan", "interest", "hobi", "reference", "referensi",
        "honors", "awards", "publication", "publikasi", "summary", "profil"
    ]
    
    start_pos = -1
    header_length = 0
    
    # 1. Cari posisi awal dari section skill
    for header in skill_headers:
        # Gunakan regex batas kata (\b) agar tidak mendeteksi substring yang salah
        match = re.search(r'\b' + re.escape(header) + r'\b', text_lower)
        if match:
            pos = match.start()
            if start_pos == -1 or pos < start_pos:
                start_pos = pos
                header_length = len(header)
            
    if start_pos == -1:
        # Jika tidak ada header skill spesifik, gunakan keseluruhan teks sebagai fallback
        return cv_text
        
    # 2. MAJUKAN titik awal pembacaan untuk MENG-EXCLUDE judul section itu sendiri
    actual_start = start_pos + header_length
    
    # 3. Cari posisi akhir (judul section berikutnya)
    end_pos = len(cv_text)
    for header in end_headers:
        match = re.search(r'\b' + re.escape(header) + r'\b', text_lower[actual_start:])
        if match:
            # Karena pencarian dilakukan di-slice [actual_start:], 
            # posisinya harus ditambah dengan actual_start
            pos = actual_start + match.start() 
            if pos < end_pos:
                end_pos = pos
            
    # Ambil teks khusus dari area skills yang sudah difilter
    extracted_section = cv_text[actual_start:end_pos].strip()
    return extracted_section

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
        experience_years = extract_experience_years(full_text)
        
        skills_only_text = extract_skills_section_only(full_text)
        cleaned_skills_text = clean_text(skills_only_text)
        print(f"\n🎯 Targeted Skills Area Extracted ({len(cleaned_skills_text)} chars)")

        # GROQ Analysis for education
        print("\n🤖 Analyzing education with Groq...")
        groq_education = analyze_education_with_groq(full_text)
        degree = groq_education.get('degree')
        major = groq_education.get('major')
        gpa = groq_education.get('gpa')
        
        # GROQ Analysis for work experiences
        print("\n🤖 Analyzing work experiences with Groq...")
        groq_analysis = analyze_cv_with_groq(full_text)
        
        # Use Groq data if available and better than regex
        groq_years = groq_analysis.get('total_years')
        groq_roles = groq_analysis.get('roles', [])
        work_experiences = groq_analysis.get('work_experiences', [])
        groq_role_groups = groq_analysis.get('role_groups', [])
        groq_error = groq_analysis.get('error')
        
        # Prefer Groq result if available
        if groq_years is not None:
            experience_years = groq_years
            print(f"✓ Using Groq analysis: {experience_years} years")
        elif experience_years:
            print(f"✓ Using regex extraction: {experience_years} years")
        else:
            print(f"✓ No work experience found")

        final_skills = set()
        raw_extracted_skills = []

        # ✅ NER-BASED SKILL EXTRACTION (with strict validation)
        if ner_pipeline:
            try:
                ner_results = ner_pipeline(cleaned_skills_text[:3500])
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
        
        # ✅ MANUAL MATCHING (regex-based, only from DB skills)
        manual_skills = extract_skills_manual(cleaned_skills_text, VALID_SKILLS_SET)

        # Combine both approaches and remove duplicates (using database skill names)
        final_skills.update(validated_ner_skills)
        final_skills.update(manual_skills)
        
        # Title-case for display; matching in match_jobs uses .lower() so casing doesn't affect logic
        final_skills_list = sorted([s.title() for s in set(final_skills)])

        print(f"✓ Skills Found (database-validated): {len(final_skills_list)}")
        if final_skills_list:
            print(f"  Samples: {', '.join(final_skills_list[:5])}")
        if validated_ner_skills:
            print(f"  NER validated skills: {', '.join(sorted(validated_ner_skills)[:5])}")
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
    Delete all expired jobs from database.
    Returns: {"status": "success", "deleted_count": int, "remaining": int, "deleted_jobs": [...]}
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        print("\n🗑️  Deleting all expired jobs...")
        deleted_jobs = []
        before_count = len(JOB_LIST)
        
        # Iterate backwards to avoid index issues when removing
        for idx in range(len(JOB_LIST) - 1, -1, -1):
            job = JOB_LIST[idx]
            if job.get('status') == 'expired':
                if delete_job_from_db(job.get('id')):
                    deleted_jobs.append({
                        "id": job.get('id'),
                        "title": job.get('job_title', 'Unknown'),
                        "company": job.get('company_name', 'Unknown'),
                        "url": job.get('job_url', '#')
                    })
                    JOB_LIST.pop(idx)
        
        after_count = len(JOB_LIST)
        deleted_count = before_count - after_count
        
        print(f"\n✅ Deletion Complete:")
        print(f"   Before: {before_count}")
        print(f"   Deleted: {deleted_count}")
        print(f"   Remaining: {after_count}")
        
        return jsonify({
            "status": "success",
            "deleted_count": deleted_count,
            "before_count": before_count,
            "remaining": after_count,
            "deleted_jobs": deleted_jobs
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
            # Return list of all skills
            skills_list = []
            for idx, skill in enumerate(sorted(VALID_SKILLS_SET), 1):
                skills_list.append({"id": idx, "name": skill})
            print(f"✅ Returned {len(skills_list)} skills")
            return jsonify(skills_list)
        
        elif request.method == 'POST':
            # Add new skill
            data = request.get_json()
            skill_name = data.get('name', '').strip()
            if skill_name:
                VALID_SKILLS_SET.add(skill_name.lower())
                print(f"✅ Added new skill: {skill_name}")
                return jsonify({"status": "success", "message": f"Skill '{skill_name}' added"}), 201
            return jsonify({"error": "Skill name required"}), 400
        
        elif request.method == 'DELETE':
            # Delete skill by ID
            skill_id = request.args.get('id')
            if skill_id:
                skills_list = sorted(list(VALID_SKILLS_SET))
                try:
                    idx = int(skill_id) - 1
                    if 0 <= idx < len(skills_list):
                        skill_to_remove = skills_list[idx]
                        VALID_SKILLS_SET.discard(skill_to_remove)
                        print(f"✅ Deleted skill: {skill_to_remove}")
                        return jsonify({"status": "success"}), 200
                except:
                    pass
            return jsonify({"error": "Skill not found"}), 404
    
    except Exception as e:
        print(f"❌ Error in manage_skills: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs', methods=['GET', 'POST', 'DELETE', 'OPTIONS'])
def manage_jobs():
    """Manage jobs database"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if request.method == 'GET':
            # PENTING: Gunakan active_only=False untuk mengambil SEMUA data
            # agar kita bisa menghitung total aktif dan expired dengan akurat
            jobs_data = load_jobs_from_db(active_only=False) 
            
            jobs_list = []
            active_count = 0
            expired_count = 0
            
            for idx, job in enumerate(jobs_data, 1):
                if job.get('status') == 'expired':
                    expired_count += 1
                    continue # Lewati baris ini agar tidak masuk ke tabel UI Admin
                
                # Jika lolos (aktif), masukkan ke daftar UI
                active_count += 1
                jobs_list.append({
                    "id": job.get('id'),
                    "list_id": active_count,
                    "title": job.get('job_title', 'Unknown'),
                    "company": job.get('company_name', 'Unknown'),
                    "url": job.get('job_url', '#'),
                    "postDate": datetime.datetime.now().isoformat(),
                    "scrapedAt": datetime.datetime.now().isoformat(),
                    "status": job.get('status', 'active')
                })

            print(f"✅ Returned {active_count} active jobs, {expired_count} expired jobs")

            return jsonify({
                "jobs": jobs_list,
                "active_count": active_count,
                "expired_count": expired_count
            })
        
        elif request.method == 'POST':
            # Upload pekerjaan: terima file CSV/Excel, proses, simpan ke DB.
            print("📥 [UPLOAD] Request POST /api/jobs diterima.")
            if 'file' not in request.files:
                print("❌ [UPLOAD] Field 'file' tidak ada di request.")
                return jsonify({"error": "Tidak ada file yang diunggah (field 'file' kosong)."}), 400

            uploaded = request.files['file']
            if not uploaded or uploaded.filename == '':
                print("❌ [UPLOAD] Nama file kosong.")
                return jsonify({"error": "Nama file kosong."}), 400

            print(f"📄 [UPLOAD] Memproses file: {uploaded.filename}")
            try:
                processed_jobs, err = process_uploaded_jobs_file(uploaded)
            except Exception as e:
                print(f"❌ [UPLOAD] Error processing uploaded file: {e}")
                return jsonify({"error": f"Gagal memproses file: {e}"}), 400

            if err:
                return jsonify({"error": err}), 400
            if not processed_jobs:
                return jsonify({"error": "Tidak ada baris pekerjaan yang valid di file."}), 400

            inserted = 0
            failed = 0
            for job in processed_jobs:
                if insert_job_to_db(job) is not None:
                    inserted += 1
                else:
                    failed += 1

            global JOB_LIST
            try:
                JOB_LIST = load_jobs_from_db()
            except Exception as e:
                print(f"⚠️ Gagal refresh JOB_LIST (data tetap tersimpan): {e}")

            print(f"✅ Upload selesai: {inserted} ditambahkan, {failed} gagal.")
            return jsonify({
                "status": "success",
                "inserted": inserted,
                "failed": failed,
                "total_processed": len(processed_jobs),
                "message": f"{inserted} pekerjaan berhasil ditambahkan ke database."
            }), 200

        elif request.method == 'DELETE':
            # Delete job by ID
            job_id = request.args.get('id')
            if job_id:
                try:
                    rowid = resolve_job_rowid(job_id)
                    if rowid is not None and delete_job_from_db(rowid):
                        print(f"✅ Deleted job rowid: {rowid}")
                        return jsonify({"status": "success"}), 200
                except Exception as e:
                    print(f"❌ Error deleting job by ID: {e}")
            return jsonify({"error": "Job not found"}), 404
    
    except Exception as e:
        print(f"❌ Error in manage_jobs: {str(e)}")
        return jsonify({"error": f"Method {request.method} tidak didukung."}), 405

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
    app.run(debug=False, port=5002, use_reloader=False)