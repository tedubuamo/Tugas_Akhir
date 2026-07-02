import datetime
import difflib
import json
import os
import re
import sqlite3

from ..config import DB_FILE

try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_process

    _HAS_RAPIDFUZZ = True
    print("✅ rapidfuzz available — using for major matching")
except ImportError:
    _HAS_RAPIDFUZZ = False
    print("⚠️  rapidfuzz not found — falling back to difflib for major matching")


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


def normalize_skill_entry(skill_str):
    if skill_str is None:
        return None

    skill = str(skill_str).strip().lower()
    if not skill:
        return None

    skill = skill.replace("&amp;", "&")
    skill = re.sub(r"\s+", " ", skill)
    skill = re.sub(r"^[\-\•\*\d\.\)\(]+", "", skill).strip()
    skill = re.sub(r"[^\w\+\#\.\-\/& ]+", " ", skill)
    skill = re.sub(r"\s+", " ", skill).strip()
    return skill or None


def is_relevant_technical_skill(normalized_skill):
    if not normalized_skill:
        return False

    irrelevant_exact = {
        "communication",
        "leadership",
        "teamwork",
        "problem solving",
        "adaptability",
        "collaboration",
        "time management",
        "negotiation",
        "presentation",
        "microsoft office",
        "english",
        "bahasa indonesia",
    }
    if normalized_skill in irrelevant_exact:
        return False

    irrelevant_patterns = [
        r"\b(motivated|passionate|hardworking|honest|responsible)\b",
        r"\b(strong communication|good communication|interpersonal)\b",
        r"\b(team player|detail oriented|fast learner)\b",
    ]
    return not any(re.search(pattern, normalized_skill) for pattern in irrelevant_patterns)


def resolve_skill_against_db(skill, skill_db):
    normalized = normalize_skill_entry(skill)
    if not normalized or not skill_db:
        return None

    if normalized in skill_db:
        return normalized

    for db_skill in skill_db:
        if db_skill.lower() == normalized:
            return db_skill
    return None


def load_programs_from_db():
    rows = fetch_db_rows("SELECT [Nama Program Studi (Inggris)] FROM program_studi")
    programs = set()
    for row in rows:
        name = row.get("Nama Program Studi (Inggris)", "")
        if name and name.strip():
            programs.add(name.strip())
    return programs


def validate_major_against_db(candidate, valid_programs, threshold=85):
    if not candidate or not valid_programs:
        return candidate

    candidate_stripped = candidate.strip()
    candidate_lower = candidate_stripped.lower()

    generic_tokens = {
        "engineering",
        "science",
        "sciences",
        "studies",
        "arts",
        "education",
        "management",
        "technology",
        "technologies",
        "systems",
        "system",
        "design",
        "applied",
        "vocational",
        "and",
        "or",
        "of",
        "in",
        "the",
        "for",
        "at",
        "to",
        "a",
        "an",
    }

    def content_tokens(text):
        return {word.lower() for word in text.split() if word.lower() not in generic_tokens}

    for program in valid_programs:
        if program.lower() == candidate_lower:
            print(f"  ✓ Major exact DB match: '{program}'")
            return program

    programs_list = list(valid_programs)
    candidate_tokens = content_tokens(candidate_stripped)

    if _HAS_RAPIDFUZZ:
        match = _rf_process.extractOne(
            candidate_stripped,
            programs_list,
            scorer=_rf_fuzz.WRatio,
        )
        if match:
            best_program, score, _ = match
            overlap = candidate_tokens.intersection(content_tokens(best_program))
            if score >= threshold and overlap:
                print(f"  ✓ Major fuzzy DB match: '{candidate_stripped}' -> '{best_program}' ({score})")
                return best_program
    else:
        best_ratio = 0
        best_program = None
        for program in programs_list:
            ratio = difflib.SequenceMatcher(None, candidate_lower, program.lower()).ratio() * 100
            overlap = candidate_tokens.intersection(content_tokens(program))
            if ratio > best_ratio and overlap:
                best_ratio = ratio
                best_program = program
        if best_program and best_ratio >= threshold:
            print(f"  ✓ Major fuzzy DB match: '{candidate_stripped}' -> '{best_program}' ({best_ratio:.1f})")
            return best_program

    return candidate


def load_skills_from_db():
    rows = fetch_db_rows("SELECT skill FROM skills WHERE skill IS NOT NULL")
    cleaned = set()
    for row in rows:
        if row.get("skill"):
            skill = normalize_skill_entry(row["skill"])
            if skill and is_relevant_technical_skill(skill):
                cleaned.add(skill)
    return cleaned


def get_data_last_updated():
    try:
        if os.path.exists(DB_FILE):
            return datetime.datetime.fromtimestamp(
                os.path.getmtime(DB_FILE),
                tz=datetime.timezone.utc,
            ).isoformat()
    except Exception as e:
        print(f"⚠️ Failed to read DB modified time: {e}")
    return None


def _normalize_job_identity_value(value):
    if value is None:
        return ""

    text = str(value).strip().lower()
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text


def _build_job_signature(job):
    normalized_url = _normalize_job_identity_value(job.get("job_url"))
    normalized_title = _normalize_job_identity_value(job.get("job_title"))
    normalized_company = _normalize_job_identity_value(job.get("company_name"))
    normalized_work_type = _normalize_job_identity_value(job.get("work_type"))
    normalized_description = _normalize_job_identity_value(job.get("description_text"))

    if normalized_url:
        return f"url::{normalized_url}"

    compact_description = normalized_description[:280] if normalized_description else ""
    return "meta::" + "||".join(
        [
            normalized_title,
            normalized_company,
            normalized_work_type,
            compact_description,
        ]
    )


def _is_preferred_job_record(candidate, current):
    candidate_active = candidate.get("status") == "active"
    current_active = current.get("status") == "active"
    if candidate_active != current_active:
        return candidate_active

    candidate_id = int(candidate.get("id") or 0)
    current_id = int(current.get("id") or 0)
    return candidate_id > current_id


def _deduplicate_jobs(jobs):
    unique_jobs = {}
    duplicate_ids = []

    for job in jobs:
        signature = _build_job_signature(job)
        existing = unique_jobs.get(signature)
        if existing is None:
            unique_jobs[signature] = job
            continue

        if _is_preferred_job_record(job, existing):
            duplicate_ids.append(existing.get("id"))
            unique_jobs[signature] = job
        else:
            duplicate_ids.append(job.get("id"))

    deduped_jobs = sorted(unique_jobs.values(), key=lambda job: int(job.get("id") or 0))
    return deduped_jobs, [job_id for job_id in duplicate_ids if job_id]


def load_jobs_from_db(active_only=True):
    rows = fetch_db_rows(
        "SELECT rowid AS id, category_main, category_sub, company_description, company_name, salary, job_url, job_title, work_type, description_text, requirements, min_experience_years, min_gpa, required_degree, required_majors, is_active FROM jobs"
    )
    jobs = []
    for row in rows:
        raw_active = row.get("is_active")
        if raw_active is None:
            active = True
        else:
            try:
                active = int(float(raw_active)) == 1
            except Exception:
                active = str(raw_active).strip().lower() in ["1", "true", "yes", "aktif"]

        row["status"] = "active" if active else "expired"
        if active_only and not active:
            continue
        jobs.append(row)

    deduped_jobs, duplicate_ids = _deduplicate_jobs(jobs)
    if duplicate_ids:
        print(f"⚠️ Duplicate jobs filtered in memory: {len(duplicate_ids)}")
    return deduped_jobs


def delete_duplicate_jobs_from_db():
    rows = fetch_db_rows(
        "SELECT rowid AS id, company_name, job_url, job_title, work_type, description_text, is_active FROM jobs"
    )
    if not rows:
        return 0

    prepared_rows = []
    for row in rows:
        raw_active = row.get("is_active")
        if raw_active is None:
            active = True
        else:
            try:
                active = int(float(raw_active)) == 1
            except Exception:
                active = str(raw_active).strip().lower() in ["1", "true", "yes", "aktif"]
        row["status"] = "active" if active else "expired"
        prepared_rows.append(row)

    _, duplicate_ids = _deduplicate_jobs(prepared_rows)
    if not duplicate_ids:
        return 0

    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.executemany("DELETE FROM jobs WHERE rowid = ?", [(job_id,) for job_id in duplicate_ids])
            conn.commit()
        print(f"✅ Removed {len(duplicate_ids)} duplicate jobs from SQLite")
        return len(duplicate_ids)
    except Exception as e:
        print(f"❌ Error deleting duplicate jobs: {e}")
        return 0


def update_job_active_state(job_id, active):
    if not job_id:
        return False
    rowcount = execute_db_change(
        "UPDATE jobs SET is_active = ? WHERE rowid = ?",
        (1 if active else 0, job_id),
    )
    return rowcount > 0


def delete_job_from_db(job_id):
    if not job_id:
        return False
    rowcount = execute_db_change("DELETE FROM jobs WHERE rowid = ?", (job_id,))
    return rowcount > 0


def resolve_job_rowid(job_identifier):
    if not job_identifier:
        return None
    try:
        return int(job_identifier)
    except Exception:
        return None


def insert_job_to_db(job):
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
                    job.get("category_main"),
                    job.get("category_sub"),
                    job.get("company_description"),
                    job.get("company_name"),
                    job.get("salary"),
                    job.get("job_url"),
                    job.get("job_title"),
                    job.get("work_type"),
                    job.get("description_text"),
                    job.get("requirements"),
                    job.get("min_experience_years"),
                    job.get("min_gpa"),
                    job.get("required_degree"),
                    job.get("required_majors"),
                    1,
                ),
            )
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        print(f"❌ Error inserting job: {e}")
        return None


def clean_html_to_text(html_text):
    if not html_text:
        return ""
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(str(html_text), "html.parser").get_text(separator=" ").strip()
    except Exception:
        return re.sub(r"<[^>]+>", " ", str(html_text)).strip()


def process_uploaded_jobs_file(file_storage):
    import pandas as pd
    from .cv_logic import extract_education_requirements

    filename = (file_storage.filename or "").lower()
    print(f"\n{'=' * 55}")
    print(f"📂 [UPLOAD] Membaca file: '{file_storage.filename}'")
    print(f"{'=' * 55}")

    if filename.endswith(".csv"):
        df = pd.read_csv(file_storage)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_storage)
    else:
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
    print(f"   Kolom ditemukan: {list(df.columns)[:8]}{'...' if len(df.columns) > 8 else ''}")

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
        raw_html = row.get(html_col) if html_col else None
        if raw_html and not isinstance(raw_html, float):
            plain = re.sub(r"<[^>]+>", " ", str(raw_html))
            plain = re.sub(r"\s+", " ", plain).strip()
        elif desc_col:
            value = row.get(desc_col)
            plain = "" if (value is None or (isinstance(value, float) and pd.isna(value))) else str(value)
        else:
            plain = ""

        edu = extract_education_requirements(plain.lower())

        def safe(colname):
            if not colname:
                return None
            value = row.get(colname)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            return value

        job_url = safe(url_col)
        if not job_url and jobid_col:
            jid = safe(jobid_col)
            if jid is not None:
                try:
                    job_url = f"https://id.jobstreet.com/id/job/{int(float(jid))}"
                except Exception:
                    job_url = None

        processed.append(
            {
                "category_main": safe(cat_main_col),
                "category_sub": safe(cat_sub_col),
                "company_description": safe(company_desc_col),
                "company_name": safe(company_col),
                "salary": safe(salary_col),
                "job_url": job_url,
                "job_title": safe(title_col),
                "work_type": safe(worktype_col),
                "description_text": plain,
                "requirements": "",
                "min_experience_years": edu.get("min_experience_years"),
                "min_gpa": edu.get("min_gpa"),
                "required_degree": edu.get("degree"),
                "required_majors": ", ".join(edu.get("majors", [])) if edu.get("majors") else "",
            }
        )

    print(f"✅ [UPLOAD] Parsing selesai: {len(processed)} baris siap dimasukkan ke DB")
    return processed, None
