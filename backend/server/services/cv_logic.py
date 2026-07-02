import datetime
import json
import os
import re
import unicodedata

import pdfplumber

from .. import state
from .data_access import (
    get_data_last_updated,
    load_jobs_from_db,
    normalize_skill_entry,
    resolve_skill_against_db,
    validate_major_against_db,
)


def clean_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_name_heuristic(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    skip = ["curriculum", "vitae", "resume", "cv", "contact", "email"]
    for line in lines[:6]:
        if 3 < len(line) < 50 and not any(token in line.lower() for token in skip) and "@" not in line:
            return line.title()
    return "Kandidat"


def extract_gpa_degree(text):
    text_lower = text.lower()
    gpa = re.search(r"(?:gpa|ipk)\s*[:]?\s*(\d[.,]\d{1,2})", text_lower)
    gpa_val = gpa.group(1).replace(",", ".") if gpa else None
    degree = None
    degrees = ["master", "bachelor", "diploma", "sarjana", "s.kom", "S.Tr. S.D.T", "s1", "s2", "d3"]
    for degree_name in degrees:
        if degree_name in text_lower:
            degree = degree_name.title()
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
}


def extract_major_from_text(text):
    text_lower = text.lower()
    for keyword, major in MAJOR_KEYWORDS.items():
        if keyword in text_lower:
            return major
    return None


def extract_work_experience_section(cv_text):
    if not cv_text:
        return ""
    patterns = [
        r"(?:work experience|pengalaman kerja|experience|employment history|professional experience)(.*?)(?:education|pendidikan|skills|keahlian|projects|proyek|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cv_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def extract_experience_from_date_ranges(cv_text):
    if not cv_text:
        return None
    year_pairs = re.findall(
        r"((?:19|20)\d{2})\s*(?:-|to|sd|s\.d)\s*((?:19|20)\d{2}|present|now|current|saat ini)",
        cv_text.lower(),
    )
    if not year_pairs:
        return None

    current_year = datetime.datetime.now().year
    total_years = 0.0
    for start_raw, end_raw in year_pairs:
        try:
            start_year = int(start_raw)
            end_year = current_year if end_raw in ["present", "now", "current", "saat ini"] else int(end_raw)
            if end_year >= start_year:
                total_years += float(end_year - start_year + 1)
        except Exception:
            continue
    return round(total_years, 1) if total_years > 0 else None


def extract_education_section(cv_text):
    if not cv_text:
        return ""
    lines = [line.strip() for line in cv_text.splitlines() if line.strip()]
    text_lower = cv_text.lower()
    patterns = [
        r"(?:education|pendidikan|academic background)(.*?)(?:experience|pengalaman|skills|keahlian|projects|proyek|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
        if match:
            start = match.start(1)
            end = match.end(1)
            section = cv_text[start:end].strip()
            if len(section) >= 40:
                return section[:2500]

    education_anchor = re.compile(
        r"\b("
        r"education|pendidikan|academic|university|universitas|college|institut|institute|"
        r"politeknik|sekolah tinggi|bachelor|master|phd|sarjana|diploma|"
        r"jurusan|major|program studi|prodi|field of study|"
        r"s\.?\s*[123]|d\.?\s*[1234]|b\.sc|bachelor'?s|master'?s"
        r")\b",
        re.IGNORECASE,
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

    return cv_text[:2500]


def analyze_cv_with_groq(cv_text):
    if not state.groq_client or not cv_text:
        return {"work_experiences": [], "total_years": None, "roles": [], "error": "Groq not available"}

    try:
        work_exp_section = extract_work_experience_section(cv_text)
        if not work_exp_section:
            print("⚠️  Work Experience section not found in CV")
            return {
                "work_experiences": [],
                "total_years": None,
                "roles": [],
                "error": "Work Experience section not found",
            }

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

        completion = state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            top_p=1,
        )
        response_text = completion.choices[0].message.content.strip()
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                return {
                    "work_experiences": [],
                    "total_years": None,
                    "roles": [],
                    "role_groups": [],
                    "error": "Could not parse Groq response",
                }
            result = json.loads(json_match.group())

        result["role_groups"] = group_work_experiences_by_role(result.get("work_experiences", []))
        print("\n🔍 GROQ Analysis Result:")
        print(f"   Work Experiences Found: {len(result.get('work_experiences', []))}")
        print(f"   Total Years: {result.get('total_years')}")
        print(f"   Roles: {', '.join(result.get('roles', []))}")
        return result
    except Exception as e:
        print(f"❌ Groq Analysis Error: {str(e)}")
        return {"work_experiences": [], "total_years": None, "roles": [], "error": str(e)}


def analyze_education_with_groq(cv_text):
    if not state.groq_client or not cv_text:
        return {**extract_education_regex(cv_text or ""), "error": "Groq not available"}

    try:
        edu_section = extract_education_section(cv_text)
        print(f"🎓 Education Section Extracted ({len(edu_section)} chars)")
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
- "PhD"
- "Master"
- "Bachelor"
- "Diploma"
- IMPORTANT: "S.Tr." or "S. Tr." (Sarjana Terapan) is always "Bachelor".
- If a university or Politeknik name is present but no explicit degree, infer "Bachelor" as default.
- If multiple university degrees, return the HIGHEST one.
- Return null ONLY if there is absolutely no university/college education at all.

Rules for "major":
- Extract the FIELD OF STUDY from the HIGHEST university degree, not institution name and not high school.
- Translate to English title case.
- If you cannot find a clear program/field name from a university, return null.

Rules for "gpa":
- Extract from the UNIVERSITY entry only.

Return ONLY the JSON object. No markdown, no explanation."""

        completion = state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
            top_p=1,
        )
        response_text = completion.choices[0].message.content.strip()
        result = json.loads(response_text)
        print(
            f"🔍 GROQ Education Result: Degree: {result.get('degree')}, Major: {result.get('major')}, GPA: {result.get('gpa')}"
        )

        regex_result = extract_education_regex(cv_text)
        if not result.get("degree") and regex_result.get("degree"):
            result["degree"] = regex_result["degree"]
            print(f"  ↳ degree filled by regex: {result['degree']}")
        if not result.get("major") and regex_result.get("major"):
            result["major"] = regex_result["major"]
            print(f"  ↳ major filled by regex: {result['major']}")
        if not result.get("gpa") and regex_result.get("gpa"):
            result["gpa"] = regex_result["gpa"]
            print(f"  ↳ gpa filled by regex: {result['gpa']}")
        return result
    except Exception as e:
        print(f"❌ Groq Education Analysis Error: {str(e)}")
        regex_result = extract_education_regex(cv_text)
        print(f"  ↳ Using regex fallback: {regex_result}")
        return {**regex_result, "error": str(e)}


def extract_experience_years(text):
    if not text:
        return None

    text_lower = text.lower()
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
            years_list.extend([int(match) if isinstance(match, str) else match for match in matches])

    if years_list:
        return float(max(years_list))

    date_pattern = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\s*(?:20|19)\d{2}\s*(?:-|to|sd|s\.d)\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\s*(?:20|19)\d{2}"
    date_matches = re.findall(date_pattern, text_lower, re.IGNORECASE)
    if date_matches:
        all_years = re.findall(r"(20|19)(\d{2})", text_lower)
        if len(all_years) >= 2:
            try:
                first_year = int(all_years[0][0] + all_years[0][1])
                last_year = int(all_years[-1][0] + all_years[-1][1])
                if last_year >= first_year:
                    return float(last_year - first_year + 1)
            except Exception:
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
                "experiences": [],
            }
        grouped[normalized_role]["total_duration_years"] += float(duration)
        grouped[normalized_role]["experiences"].append(
            {
                "company": exp.get("company"),
                "position": role,
                "start_year": exp.get("start_year"),
                "end_year": exp.get("end_year"),
                "duration_years": float(duration),
            }
        )
    for group in grouped.values():
        group["total_duration_years"] = round(group["total_duration_years"], 1)
    return list(grouped.values())


def normalize_skill_variants(skill_str):
    if not skill_str:
        return skill_str
    skill_lower = skill_str.lower().strip()
    skill_normalization = {
        "deliver": "delivery",
        "delivered": "delivery",
        "delivering": "delivery",
        "test": "testing",
        "tested": "testing",
        "tests": "testing",
        "manage": "management",
        "managed": "management",
        "managing": "management",
        "manager": "management",
        "develop": "development",
        "developed": "development",
        "developing": "development",
        "developer": "development",
        "design": "design",
        "designed": "design",
        "designing": "design",
        "designer": "design",
        "analyze": "analysis",
        "analyzed": "analysis",
        "analyzing": "analysis",
        "analyst": "analysis",
        "optimize": "optimization",
        "optimized": "optimization",
        "optimizing": "optimization",
        "implement": "implementation",
        "implemented": "implementation",
        "implementing": "implementation",
        "monitor": "monitoring",
        "monitored": "monitoring",
        "monitoring": "monitoring",
        "report": "reporting",
        "reported": "reporting",
        "reporting": "reporting",
        "document": "documentation",
        "documented": "documentation",
        "documenting": "documentation",
        "visualize": "visualization",
        "visualized": "visualization",
        "visualizing": "visualization",
        "support": "support",
        "supported": "support",
        "supporting": "support",
        "lead": "leadership",
        "leading": "leadership",
        "led": "leadership",
        "train": "training",
        "trained": "training",
        "training": "training",
        "plan": "planning",
        "planned": "planning",
        "planning": "planning",
        "improve": "improvement",
        "improved": "improvement",
        "improving": "improvement",
        "integrate": "integration",
        "integrated": "integration",
        "integrating": "integration",
        "deploy": "deployment",
        "deployed": "deployment",
        "deploying": "deployment",
        "automate": "automation",
        "automated": "automation",
        "automating": "automation",
        "create": "creation",
        "created": "creation",
        "creating": "creation",
        "build": "building",
        "built": "building",
        "building": "building",
        "collaborate": "collaboration",
        "collaborated": "collaboration",
        "collaborating": "collaboration",
        "communicate": "communication",
        "communicated": "communication",
        "communicating": "communication",
        "present": "presentation",
        "presented": "presentation",
        "presenting": "presentation",
        "organize": "organization",
        "organized": "organization",
        "organizing": "organization",
    }
    canonical = skill_normalization.get(skill_lower)
    if canonical:
        return canonical.title()
    return skill_str


def deduplicate_normalized_skills(skills_list):
    if not skills_list:
        return []
    normalized_map = {}
    for skill in skills_list:
        normalized = normalize_skill_variants(skill)
        normalized_lower = normalized.lower()
        if normalized_lower not in normalized_map:
            normalized_map[normalized_lower] = normalized
    return sorted(list(normalized_map.values()))


def is_valid_skill_string(skill_str):
    normalized = normalize_skill_entry(skill_str)
    if not normalized:
        return False
    cleaned = str(skill_str).strip()
    months = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}
    if normalized in months:
        return False
    allowed_short_tokens = {"ai", "bi", "ci", "ml", "nlp", "sql", "aws", "gcp", "api", "etl", "elt", "ui", "ux"}
    if len(normalized) <= 3 and cleaned.isupper() and normalized not in allowed_short_tokens:
        return False
    return True


def sanitize_requirement_skill_list(skills):
    if not skills:
        return []
    sanitized = []
    seen = set()
    valid_skill_map = {s.lower(): s for s in state.VALID_SKILLS_SET}
    for raw_skill in skills:
        normalized = normalize_skill_entry(raw_skill)
        if not normalized or normalized not in valid_skill_map:
            continue
        canonical = valid_skill_map[normalized]
        if canonical not in seen:
            seen.add(canonical)
            sanitized.append(canonical)
    return sanitized


def parse_requirement_skills(requirements_text):
    if not requirements_text:
        return []
    raw_items = [item.strip() for item in str(requirements_text).split(",") if item.strip()]
    return sanitize_requirement_skill_list(raw_items)


def validate_skills_against_db(skills, skill_db):
    validated = set()
    if not skills:
        return validated
    for skill in skills:
        if not skill or not is_valid_skill_string(skill):
            continue
        db_skill = resolve_skill_against_db(skill, skill_db)
        if db_skill:
            validated.add(db_skill)
    return validated


def extract_skills_manual(text, skill_db):
    if not text or not skill_db:
        return []
    text_lower = text.lower()
    skill_set = skill_db if isinstance(skill_db, set) else set(skill_db)
    tokens = re.split(r"[\s,;:|()\[\]{}'\"]+", text_lower)
    tokens = [token.strip(".").strip() for token in tokens if token.strip() and len(token.strip()) >= 1]
    found = set()
    max_ngram = 5
    for n in range(1, max_ngram + 1):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            if ngram in skill_set and is_valid_skill_string(ngram):
                found.add(ngram)
    return list(found)


def extract_education_requirements(text):
    if not text:
        return {"degree": None, "majors": [], "min_gpa": None, "min_experience_years": None}
    text_lower = text.lower()
    result = {"degree": None, "majors": [], "min_gpa": None, "min_experience_years": None}

    degree_patterns = [
        (r"master|s2|s\.2|pascasarjana", "Master"),
        (r"bachelor|s1|s\.1|sarjana|diploma|d3|d4", "Bachelor"),
        (r"diploma|d3", "Diploma"),
    ]
    for pattern, degree in degree_patterns:
        if re.search(pattern, text_lower):
            result["degree"] = degree
            break

    gpa_match = re.search(r"gpa|ipk[:\s]*(\d[.,]\d{1,2})", text_lower)
    if gpa_match:
        try:
            result["min_gpa"] = float(gpa_match.group(1).replace(",", "."))
        except Exception:
            pass

    exp_patterns = [
        r"(?:minimum\s+)?(?:pengalaman|experience)\s*[:]?\s*(\d+)\s*(?:\+|tahun|years?|yrs)",
        r"(\d+)\s*(?:\+|tahun|years?|yrs)\s*(?:pengalaman|experience)",
        r"(?:exp|experience)\s*[:]?\s*(\d+)\s*(?:years?|tahun)",
        r"fresh\s*graduate",
    ]
    for pattern in exp_patterns:
        if "fresh" in pattern and "fresh graduate" in text_lower:
            result["min_experience_years"] = 0
            break
        exp_match = re.search(pattern, text_lower)
        if exp_match:
            try:
                result["min_experience_years"] = float(exp_match.group(1))
                break
            except Exception:
                pass

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
    if not raw:
        return None
    degree = raw.lower().strip()
    if degree in ("s3", "phd", "ph.d", "doktor", "doctorate", "doctor"):
        return "phd"
    if degree in ("s2", "magister", "master", "master's", "m.sc", "m.s", "m.t", "m.kom"):
        return "master"
    if degree in ("s1", "d4", "sarjana", "bachelor", "bachelor's", "bachelors"):
        return "bachelor"
    if degree in ("d3", "d2", "d1", "diploma", "ahli madya", "a.md", "amd"):
        return "diploma"
    return degree


def _parse_majors(raw):
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, TypeError):
            pass
    return [item.strip() for item in text.split(",") if item.strip()]


def calculate_compatibility_score(user_skills, job, user_degree=None, user_gpa=None, user_major=None, user_experience_years=None):
    skill_weight = 0.64
    experience_weight = 0.25
    education_weight = 0.11
    score_components = {}
    degree_hierarchy = {"diploma": 1, "bachelor": 2, "master": 3, "phd": 4, "doctorate": 4}

    req_list = parse_requirement_skills(job.get("requirements", ""))
    user_set = set(skill.lower() for skill in user_skills)
    job_set = set(req.lower() for req in req_list)
    matches = user_set.intersection(job_set)
    match_count = len(matches)
    total_reqs = len(job_set)
    skill_score = (match_count / total_reqs) * 100 if total_reqs > 0 else 0
    score_components["skill_score"] = skill_score

    edu_req_text = extract_education_requirements(job.get("description_text", ""))
    req_degree_raw = job.get("required_degree") or edu_req_text.get("degree")
    req_degree = req_degree_raw
    req_majors = _parse_majors(job.get("required_majors")) or edu_req_text.get("majors") or []
    req_min_gpa = job.get("min_gpa") or edu_req_text.get("min_gpa")

    education_score = 0
    education_status_list = []
    if not req_degree:
        education_score += 60
        education_status_list.append("✓ Tidak ada persyaratan gelar khusus")
    else:
        user_norm = _normalize_degree(user_degree)
        req_norm = _normalize_degree(req_degree)
        user_lvl = degree_hierarchy.get(user_norm, 0) if user_norm else 0
        req_lvl = degree_hierarchy.get(req_norm, 0)
        if user_degree:
            if req_lvl == 0:
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

    if not req_majors:
        education_score += 40
        education_status_list.append("✓ Tidak ada persyaratan jurusan khusus")
    else:
        if user_major:
            user_major_lower = user_major.lower()
            req_majors_lower = [major.lower() for major in req_majors]
            if any(user_major_lower == required or user_major_lower in required or required in user_major_lower for required in req_majors_lower):
                education_score += 40
                education_status_list.append(f"✓ Jurusan cocok: {user_major}")
            else:
                education_score += 10
                education_status_list.append(f"~ Jurusan berbeda: {user_major} (syarat: {', '.join(req_majors)})")
        else:
            education_score += 20
            education_status_list.append(f"~ Jurusan tidak terdeteksi (syarat: {', '.join(req_majors)})")

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

    score_components["education_score"] = min(100, education_score)
    score_components["education_status"] = " | ".join(education_status_list) if education_status_list else "N/A"

    min_exp_years = job.get("min_experience_years")
    if min_exp_years is None or min_exp_years == "":
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

    score_components["experience_score"] = experience_score
    score_components["experience_status"] = experience_status

    score_components["weights"] = {
        "skill": skill_weight,
        "experience": experience_weight,
        "education": education_weight,
    }
    score_components["weighted_scores"] = {
        "skill": round(skill_score * skill_weight, 1),
        "experience": round(experience_score * experience_weight, 1),
        "education": round(score_components["education_score"] * education_weight, 1),
    }

    total_score = (
        (skill_score * skill_weight)
        + (experience_score * experience_weight)
        + (score_components["education_score"] * education_weight)
    )
    return round(total_score, 1), score_components


def match_jobs(user_skills, jobs, user_degree=None, user_gpa=None, user_major=None, user_experience_years=None, user_roles=None):
    if not user_skills or not jobs:
        return []
    user_set = set(skill.lower() for skill in user_skills)
    user_roles_set = set(role.lower() for role in (user_roles or [])) if user_roles else set()
    user_skill_map = {skill.lower(): skill for skill in user_skills}
    ranked = []

    for job in jobs:
        req_list = parse_requirement_skills(job.get("requirements", ""))
        if not req_list:
            continue
        job_set = set(req.lower() for req in req_list if req.strip())
        matches = user_set.intersection(job_set)
        match_count = len(matches)
        total_reqs = len(job_set)
        if total_reqs <= 0:
            continue

        skill_score = (match_count / total_reqs) * 100
        if skill_score <= 0:
            continue

        job_title_lower = job.get("job_title", "").lower()
        job_description_lower = job.get("description_text", "").lower()
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

        edu_req = extract_education_requirements(job.get("description_text", ""))
        min_exp_years = job.get("min_experience_years")
        compatibility_score, score_components = calculate_compatibility_score(
            user_skills,
            job,
            user_degree,
            user_gpa,
            user_major,
            user_experience_years,
        )
        adjusted_compatibility = min(100, compatibility_score + role_match_bonus)

        ranked.append(
            {
                "title": job.get("job_title", "Unknown"),
                "company": job.get("company_name", "Unknown"),
                "match_score": round(skill_score, 1),
                "compatibility_score": adjusted_compatibility,
                "role_match_bonus": role_match_bonus,
                "role_match_details": role_match_details,
                "skill_score": round(score_components.get("skill_score", 0), 1),
                "experience_score": round(score_components.get("experience_score", 0), 1),
                "education_score": round(score_components.get("education_score", 0), 1),
                "score_weights": score_components.get("weights", {}),
                "weighted_scores": score_components.get("weighted_scores", {}),
                "matched_skills": [user_skill_map.get(match, match.title()) for match in matches],
                "missing_skills": [missing.title() for missing in (job_set - user_set)],
                "job_url": job.get("job_url", "#"),
                "min_experience_years": min_exp_years,
                "user_experience_match": score_components.get("experience_status", "N/A"),
                "education_required": {
                    "degree": job.get("required_degree") or edu_req.get("degree"),
                    "majors": _parse_majors(job.get("required_majors")) or edu_req.get("majors", []),
                    "min_gpa": job.get("min_gpa") or edu_req.get("min_gpa"),
                    "min_experience_years": edu_req.get("min_experience_years"),
                },
            }
        )

    ranked.sort(key=lambda item: item["compatibility_score"], reverse=True)
    return ranked[:50]


def extract_cv_section_blocks(cv_text, start_headers, end_headers):
    if not cv_text:
        return []
    text_lower = cv_text.lower()
    collected = []
    for header in start_headers:
        match = re.search(r"(?:^|\n)[ \t]*" + re.escape(header) + r"[ \t]*(?:\n|:|$)", text_lower, re.MULTILINE)
        if not match:
            continue
        actual_start = match.end()
        end_pos = len(cv_text)
        for end_header in end_headers:
            end_match = re.search(
                r"(?:^|\n)[ \t]*" + re.escape(end_header) + r"[ \t]*(?:\n|:|$)",
                text_lower[actual_start:],
                re.MULTILINE,
            )
            if end_match:
                pos = actual_start + end_match.start()
                if pos < end_pos:
                    end_pos = pos
        snippet = cv_text[actual_start:end_pos].strip()
        if snippet:
            collected.append(snippet)

    seen = set()
    ordered = []
    for block in collected:
        key = block.strip().lower()
        if key and key not in seen:
            seen.add(key)
            ordered.append(block)
    return ordered


def extract_targeted_skill_context(cv_text):
    if not cv_text:
        return ""
    skill_headers = ["technical skills", "core competencies", "technologies", "skills", "keahlian", "kemampuan", "expertise", "tools"]
    summary_headers = ["summary", "professional summary", "profile", "profil", "ringkasan", "about me", "objective"]
    project_headers = ["project", "projects", "proyek", "portfolio", "portofolio", "selected projects", "personal projects"]
    end_headers = [
        "experience",
        "pengalaman",
        "education",
        "pendidikan",
        "certifications",
        "sertifikasi",
        "language",
        "bahasa",
        "work history",
        "achievement",
        "penghargaan",
        "interest",
        "hobi",
        "reference",
        "referensi",
        "honors",
        "awards",
        "publication",
        "publikasi",
    ]
    blocks = []
    blocks.extend(extract_cv_section_blocks(cv_text, summary_headers, end_headers))
    blocks.extend(extract_cv_section_blocks(cv_text, project_headers, end_headers))
    blocks.extend(extract_cv_section_blocks(cv_text, skill_headers, end_headers))
    if blocks:
        return "\n".join(blocks)
    return cv_text[:2500]


def extract_education_regex(cv_text):
    if not cv_text:
        return {"degree": None, "major": None, "gpa": None}
    text = cv_text
    degree = None
    degree_patterns = [
        (r"\b(S\.?\s*3|Doktor|Ph\.?\s*D\.?|Doctor(?:ate)?)\b", "PhD"),
        (r"\b(S\.?\s*2|Magister|M\.Sc\.?|M\.S\.?|M\.Eng\.?|M\.Kom|M\.T\.?|M\.Si|M\.M|M\.Pd|Master(?:'?s)?)\b", "Master"),
        (r"\bS\.\s*(?:ST|Kom|T|Ds|E|Psi|Sos|H|Ked|Hut|Pi|IP|Farm|Si)\b", "Bachelor"),
        (r"\bS\.?\s*Tr\.?(?:\s*\.\s*\w+)?", "Bachelor"),
        (r"\bB\.\s*(?:S(?:c)?|E(?:ng)?|Tech|A|Com)\.?\b", "Bachelor"),
        (r"\b(S\.?\s*1|D\.?\s*4|Sarjana|Bachelor(?:'?s)?(?:\s+(?:of|degree|in))?|Undergraduate)\b", "Bachelor"),
        (r"\b(D\.?\s*3|D\.?\s*2|D\.?\s*1|A\.?Md\.?|Ahli\s+Madya|Diploma(?:\s+\d)?|Associate(?:'?s)?)\b", "Diploma"),
    ]
    for pattern, label in degree_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            degree = label
            break

    gpa = None
    gpa_match = re.search(r"(?:IPK|GPA|IP Kumulatif|Indeks Prestasi)[^0-9]*(\d+[.,]\d+)", text, re.IGNORECASE)
    if not gpa_match:
        gpa_match = re.search(r"\b([3-4]\.\d{2})\s*/\s*4(?:\.00)?\b", text)
    if gpa_match:
        try:
            gpa = float(gpa_match.group(1).replace(",", "."))
        except Exception:
            pass

    major_map = [
        (r"\bsains\s+data\s+terapan\b|\bapplied\s+data\s+science\b", "Applied Data Science"),
        (r"\bdata\s+science\b|\bdata\s+sains\b|\bsains\s+data\b|\bilmu\s+data\b", "Data Science"),
        (r"\bteknik\s+informatika\b|\bcomputer\s+science\b|\bilmu\s+komputer\b", "Computer Science"),
        (r"\bsistem\s+informasi\b|\binformation\s+systems?\b", "Information Systems"),
        (r"\bteknologi\s+informasi\b|\binformation\s+technology\b", "Information Technology"),
        (r"\bkecerdasan\s+buatan\b|\bartificial\s+intelligence\b", "Artificial Intelligence"),
        (r"\brekayasa\s+perangkat\s+lunak\b|\bsoftware\s+engineering\b", "Software Engineering"),
        (r"\bteknik\s+komputer\b|\bcomputer\s+engineering\b", "Computer Engineering"),
        (r"\bteknik\s+elektro\b|\belectrical\s+engineering\b", "Electrical Engineering"),
        (r"\bteknik\s+industri\b|\bindustrial\s+engineering\b", "Industrial Engineering"),
        (r"\bteknik\s+mesin\b|\bmechanical\s+engineering\b", "Mechanical Engineering"),
        (r"\bteknik\s+sipil\b|\bcivil\s+engineering\b", "Civil Engineering"),
        (r"\bmanajemen\s+informatika\b|\binformatics\s+management\b", "Informatics Management"),
        (r"\bmanajemen\s+bisnis\b|\bbusiness\s+management\b", "Business Management"),
        (r"\bmanajemen\b|\bmanagement\b", "Management"),
        (r"\bakuntansi\b|\baccounting\b", "Accounting"),
        (r"\bekonomi\b|\beconomics\b", "Economics"),
        (r"\bstatistika\b|\bstatistics\b|\bstatistik\b", "Statistics"),
        (r"\bmatematika\b|\bmathematics\b", "Mathematics"),
        (r"\bfisika\b|\bphysics\b", "Physics"),
        (r"\bkomunikasi\b|\bcommunications?\b", "Communications"),
        (r"\bpsikologi\b|\bpsychology\b", "Psychology"),
        (r"\bhukum\b|\blaw\b", "Law"),
        (r"\bkedokteran\b|\bmedicine\b|\bmedical\b", "Medicine"),
        (r"\bkeperawatan\b|\bnursing\b", "Nursing"),
        (r"\bdesain\s+komunikasi\s+visual\b|\bvisual\s+communication\b", "Visual Communication Design"),
        (r"\bdesain\s+grafis\b|\bgraphic\s+design\b", "Graphic Design"),
        (r"\barsitektur\b|\barchitecture\b", "Architecture"),
    ]
    explicit_major_patterns = [
        r"(?:major|jurusan|program studi|prodi|field of study)\s*[:\-]?\s*([A-Za-z&.,/() \-]{3,120})",
        r"(?:bachelor|master|diploma|sarjana|s\.?\s*[123]|d\.?\s*[1234])[^.\n]{0,80}?\bin\s+([A-Za-z&.,/() \-]{3,120})",
    ]

    def pick_best_major(source_text):
        if not source_text:
            return None
        candidates = []
        source_lines = [line.strip() for line in source_text.splitlines() if line.strip()]
        proximity_hint = re.compile(
            r"\b(university|universitas|college|institut|institute|politeknik|sekolah tinggi|"
            r"bachelor|master|phd|sarjana|diploma|s\.?\s*[123]|d\.?\s*[1234])\b",
            re.IGNORECASE,
        )
        for idx, line in enumerate(source_lines):
            base_score = 0
            if re.search(r"\b(major|jurusan|program studi|prodi|field of study)\b", line, re.IGNORECASE):
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
                    candidates.append((base_score + len(match.group(0)), idx, label))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]

    edu_section = extract_education_section(text)
    major = pick_best_major(edu_section)
    if not major:
        education_like_lines = []
        line_anchor = re.compile(
            r"\b(education|pendidikan|academic|university|universitas|college|institut|institute|"
            r"politeknik|sekolah tinggi|major|jurusan|program studi|prodi|field of study|"
            r"bachelor|master|phd|sarjana|diploma|s\.?\s*[123]|d\.?\s*[1234])\b",
            re.IGNORECASE,
        )
        for raw_line in text.splitlines():
            if line_anchor.search(raw_line):
                education_like_lines.append(raw_line)
        major = pick_best_major("\n".join(education_like_lines))
    return {"degree": degree, "major": major, "gpa": gpa}


def resolve_major_conflict(cv_text, groq_major=None, regex_major=None):
    if not groq_major:
        return regex_major
    if not regex_major:
        return groq_major
    if groq_major.strip().lower() == regex_major.strip().lower():
        return groq_major

    edu_section = extract_education_section(cv_text or "")
    edu_lower = edu_section.lower()
    alias_map = {
        "Applied Data Science": [r"\bapplied\s+data\s+science\b", r"\bsains\s+data\s+terapan\b"],
        "Data Science": [r"\bdata\s+science\b", r"\bdata\s+sains\b", r"\bsains\s+data\b", r"\bilmu\s+data\b"],
        "Computer Science": [r"\bcomputer\s+science\b", r"\bteknik\s+informatika\b", r"\bilmu\s+komputer\b"],
        "Computer Science or Informatics": [r"\bcomputer\s+science\b", r"\binformatics\b", r"\bteknik\s+informatika\b", r"\bilmu\s+komputer\b"],
        "Informatics Engineering": [r"\binformatics\s+engineering\b", r"\bteknik\s+informatika\b"],
        "Information Systems": [r"\binformation\s+systems?\b", r"\bsistem\s+informasi\b"],
        "Information Technology": [r"\binformation\s+technology\b", r"\bteknologi\s+informasi\b"],
    }

    def supported_by_education(major_name):
        patterns = alias_map.get(major_name, [r"\b" + re.escape(major_name.lower()) + r"\b"])
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
    if regex_major not in generic_it_family and groq_major in generic_it_family:
        print(f"  ↳ major conflict resolved by specificity: {regex_major} over {groq_major}")
        return regex_major
    return groq_major


def extract_skills_section_only(cv_text):
    if not cv_text:
        return ""
    skill_headers = ["technical skills", "core competencies", "technologies", "skills", "keahlian", "kemampuan", "expertise", "tools"]
    end_headers = [
        "experience",
        "pengalaman",
        "education",
        "pendidikan",
        "project",
        "proyek",
        "certifications",
        "sertifikasi",
        "language",
        "bahasa",
        "work history",
        "achievement",
        "penghargaan",
        "interest",
        "hobi",
        "reference",
        "referensi",
        "honors",
        "awards",
        "publication",
        "publikasi",
        "summary",
        "profil",
    ]
    blocks = extract_cv_section_blocks(cv_text, skill_headers, end_headers)
    return "\n".join(blocks)


def extract_structured_skill_candidates(skills_section_text):
    if not skills_section_text:
        return []
    candidates = []
    lines = [line.strip() for line in skills_section_text.splitlines() if line.strip()]
    for line in lines:
        if ":" in line:
            line = line.split(":", 1)[1]
        parts = re.split(r"[,|;/]\s*", line)
        for part in parts:
            candidate = re.sub(r"^[\-\*\u2022]\s*", "", part).strip()
            if candidate and len(candidate) <= 80:
                candidates.append(candidate)
    return candidates


def build_cv_analysis_response(file_storage):
    full_text = ""
    with pdfplumber.open(file_storage) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"

    if not full_text.strip():
        raise ValueError("Tidak ada teks yang dapat diekstrak dari PDF")

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

    print("\n🤖 Analyzing education...")
    regex_edu = extract_education_regex(full_text)
    groq_education = analyze_education_with_groq(full_text)
    degree = groq_education.get("degree") or regex_edu.get("degree")
    major = resolve_major_conflict(full_text, groq_education.get("major"), regex_edu.get("major"))
    gpa = groq_education.get("gpa") or regex_edu.get("gpa")

    if not degree:
        broad_degree_patterns = [
            (r"\bS\.?\s*Tr\.?(?:\s*\.\s*\w+)?", "Bachelor"),
            (r"\bB\.\s*(?:S(?:c)?|E(?:ng)?|Tech|A|Com)\.?\b", "Bachelor"),
            (r"\b(?:Bachelor|Sarjana|S\.?\s*1|D\.?\s*4|Undergraduate)\b", "Bachelor"),
            (r"\bS\.\s*(?:ST|Kom|T|Ds|E|Psi|Sos|H|Farm|Si)\b", "Bachelor"),
            (r"\b(?:Master|Magister|S\.?\s*2|M\.Sc|M\.S\.?|M\.Eng)\b", "Master"),
            (r"\b(?:Ph\.?\s*D\.?|Doktor|S\.?\s*3)\b", "PhD"),
            (r"\b(?:D\.?\s*[123]|A\.?Md\.?|Ahli\s+Madya|Associate)\b", "Diploma"),
            (r"\b(?:Universitas|University|Institut\b|Institute|College|Sekolah\s+Tinggi|Politeknik)\b", "Bachelor"),
        ]
        for pattern, label in broad_degree_patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                degree = label
                print(f"  ↳ degree filled by broad scan: {degree}")
                break

    if major:
        print(f"\n🔍 Validating major '{major}' against program_studi DB...")
        major = validate_major_against_db(major, state.VALID_PROGRAMS_SET)

    print(f"✓ Education: degree={degree}, major={major}, gpa={gpa}")
    print("\n🤖 Analyzing work experiences with Groq...")
    groq_analysis = analyze_cv_with_groq(full_text)
    groq_years = groq_analysis.get("total_years")
    groq_roles = groq_analysis.get("roles") or []
    work_experiences = groq_analysis.get("work_experiences") or []
    groq_role_groups = groq_analysis.get("role_groups") or []
    groq_error = groq_analysis.get("error")

    if groq_years is not None:
        experience_years = groq_years
        print(f"✓ Using Groq analysis: {experience_years} years")
    elif experience_years:
        print(f"✓ Using regex extraction: {experience_years} years")
    else:
        date_range_years = extract_experience_from_date_ranges(work_exp_section) if work_exp_section else None
        if date_range_years:
            experience_years = date_range_years
            print(f"✓ Using date-range accumulation: {experience_years} years")
        else:
            print("✓ No work experience found")

    final_skills = set()
    raw_extracted_skills = []
    if state.ner_pipeline:
        try:
            ner_input_text = cleaned_skills_text or cleaned_targeted_skill_context
            ner_results = state.ner_pipeline(ner_input_text[:3500])
            for item in ner_results:
                if item["entity_group"] == "SKILL":
                    word = re.sub(r"[^\w\+\#\.\-]", "", item["word"]).strip()
                    if word and len(word) >= 2 and is_valid_skill_string(word):
                        raw_extracted_skills.append(word)
        except Exception as e:
            print(f"⚠️  NER Pipeline error: {e}")

    validated_ner_skills = validate_skills_against_db(raw_extracted_skills, state.VALID_SKILLS_SET)
    structured_skill_candidates = extract_structured_skill_candidates(skills_only_text)
    structured_skills = validate_skills_against_db(structured_skill_candidates, state.VALID_SKILLS_SET)
    manual_skills = extract_skills_manual(
        cleaned_targeted_skill_context or cleaned_skills_text,
        state.VALID_SKILLS_SET,
    )

    final_skills.update(validated_ner_skills)
    final_skills.update(structured_skills)
    final_skills.update(manual_skills)
    final_skills_list = sorted([skill.title() for skill in set(final_skills)])

    print(f"✓ Skills Found (database-validated): {len(final_skills_list)}")
    if final_skills_list:
        print(f"  Samples: {', '.join(final_skills_list[:5])}")

    active_job_list = load_jobs_from_db(active_only=True)
    recs = match_jobs(
        final_skills_list,
        active_job_list,
        user_degree=degree,
        user_gpa=gpa,
        user_major=major,
        user_experience_years=experience_years,
        user_roles=groq_roles,
    )

    all_missing_skills = set()
    all_recommended_skills = set()
    user_skills_lower = {skill.lower() for skill in final_skills_list}
    db_skills_lower = {skill.lower() for skill in state.VALID_SKILLS_SET}
    for job in recs:
        for skill in job.get("missing_skills") or []:
            skill_lower = skill.lower()
            if skill_lower in db_skills_lower:
                all_missing_skills.add(skill_lower)
        for skill in job.get("matched_skills") or []:
            skill_lower = skill.lower()
            if skill_lower in db_skills_lower and skill_lower not in user_skills_lower:
                all_recommended_skills.add(skill_lower)

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
            "work_experiences": work_experiences[:5],
        },
        "groq_analysis": {
            "roles": groq_roles,
            "work_experiences": work_experiences,
            "role_groups": groq_role_groups,
            "total_years": groq_years,
            "error": groq_error,
        },
        "data_last_updated": get_data_last_updated(),
        "skills_detected": final_skills_list,
        "skills_missing": sorted([skill.title() for skill in all_missing_skills]),
        "skills_recommended": sorted([skill.title() for skill in all_recommended_skills]),
        "job_recommendations": recs,
    }
    print(f"\n✅ Response prepared: {len(final_skills_list)} skills, {len(recs)} job recommendations")
    return response


def set_groq_api_key(api_key):
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("API key cannot be empty")
    from groq import Groq

    test_client = Groq(api_key=api_key)
    test_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=10,
        messages=[{"role": "user", "content": "Test"}],
    )
    state.groq_client = test_client
    os.environ["GROQ_API_KEY"] = api_key
    print("✅ Groq API Key Set Successfully!")
