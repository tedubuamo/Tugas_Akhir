import threading
import time as _time
import traceback

from flask import Blueprint, jsonify, request

from .. import state
from ..services.cv_logic import (
    build_cv_analysis_response,
    extract_skills_manual,
    sanitize_requirement_skill_list,
    set_groq_api_key,
)
from ..services.data_access import (
    delete_duplicate_jobs_from_db,
    delete_job_from_db,
    execute_db_change,
    fetch_db_rows,
    get_data_last_updated,
    is_relevant_technical_skill,
    load_jobs_from_db,
    normalize_skill_entry,
    process_uploaded_jobs_file,
    resolve_job_rowid,
    update_job_active_state,
    insert_job_to_db,
)
from ..services.link_checker import get_link_check_snapshot, start_link_check


api_bp = Blueprint("api", __name__)


@api_bp.route("/extract-cv/", methods=["POST", "OPTIONS"])
def extract_cv():
    if request.method == "OPTIONS":
        return "", 204
    if "pdf" not in request.files:
        print("❌ ERROR: No file provided")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["pdf"]
    if not file or file.filename == "":
        print("❌ ERROR: File is empty")
        return jsonify({"error": "File is empty"}), 400

    print(f"\n📄 PROSES FILE: {file.filename}")
    try:
        return jsonify(build_cv_analysis_response(file))
    except ValueError as e:
        print(f"❌ ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500


@api_bp.route("/set-groq-key/", methods=["POST"])
def set_groq_key():
    data = request.get_json()
    if not data or "api_key" not in data:
        return jsonify({"error": "Missing api_key in request"}), 400
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "API key cannot be empty"}), 400
    try:
        set_groq_api_key(api_key)
        return jsonify(
            {
                "status": "success",
                "message": "Groq API Key configured successfully",
                "groq_initialized": True,
            }
        )
    except Exception as e:
        print(f"❌ Invalid Groq API Key: {str(e)}")
        return jsonify(
            {
                "status": "error",
                "message": f"Failed to validate API key: {str(e)}",
                "groq_initialized": False,
            }
        ), 400


@api_bp.route("/api/check-links", methods=["POST", "OPTIONS"])
def check_links():
    if request.method == "OPTIONS":
        return "", 204
    if not start_link_check():
        return (
            jsonify(
                {
                    "status": "already_running",
                    "message": "Pengecekan sedang berjalan.",
                    "total": state.CHECK_STATE.get("total", 0),
                    "checked": state.CHECK_STATE.get("checked", 0),
                }
            ),
            200,
        )
    return jsonify({"status": "started", "message": "Pengecekan link dimulai di latar belakang."}), 202


@api_bp.route("/api/check-links/status", methods=["GET", "OPTIONS"])
def check_links_status():
    if request.method == "OPTIONS":
        return "", 204
    snapshot = get_link_check_snapshot()
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


@api_bp.route("/api/delete-expired", methods=["POST", "OPTIONS"])
def delete_expired():
    if request.method == "OPTIONS":
        return "", 204
    try:
        print("\n🛑 Marking expired jobs as deactivated...")
        all_jobs = load_jobs_from_db(active_only=False)
        expired = [job for job in all_jobs if job.get("status") == "expired"]
        deactivated_jobs = []
        for job in expired:
            if update_job_active_state(job.get("id"), False):
                deactivated_jobs.append(
                    {
                        "id": job.get("id"),
                        "title": job.get("job_title", "Unknown"),
                        "company": job.get("company_name", "Unknown"),
                        "url": job.get("job_url", "#"),
                    }
                )
        deactivated_count = len(deactivated_jobs)
        state.JOB_LIST = load_jobs_from_db(active_only=True)
        active_count = len(state.JOB_LIST)
        expired_count = len(load_jobs_from_db(active_only=False)) - active_count
        total_count = active_count + expired_count
        print(f"✅ Marked {deactivated_count} expired jobs as deactivated ({active_count} active, {expired_count} expired)")
        return jsonify(
            {
                "status": "success",
                "deactivated_count": deactivated_count,
                "expired_count": expired_count,
                "active_count": active_count,
                "total_count": total_count,
                "deactivated_jobs": deactivated_jobs,
            }
        )
    except Exception as e:
        print(f"❌ Error in delete_expired: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@api_bp.route("/api/skills", methods=["GET", "POST", "DELETE", "OPTIONS"])
def manage_skills():
    if request.method == "OPTIONS":
        return "", 204
    try:
        if request.method == "GET":
            rows = fetch_db_rows("SELECT rowid AS id, skill AS name FROM skills ORDER BY skill")
            filtered_rows = []
            for row in rows:
                normalized = normalize_skill_entry(row.get("name"))
                if normalized and is_relevant_technical_skill(normalized):
                    filtered_rows.append({"id": row.get("id"), "name": row.get("name")})
            print(f"✅ Returned {len(filtered_rows)} filtered skills from DB")
            return jsonify(filtered_rows)

        if request.method == "POST":
            data = request.get_json()
            skill_name = (data.get("name") or "").strip()
            if not skill_name:
                return jsonify({"error": "Skill name required"}), 400
            skill_lower = normalize_skill_entry(skill_name)
            if not skill_lower:
                return jsonify({"error": "Skill tidak valid atau tidak bermakna untuk database skill"}), 400
            if not is_relevant_technical_skill(skill_lower):
                return jsonify({"error": "Skill tidak relevan untuk domain IT / Data Science"}), 400
            if skill_lower in state.VALID_SKILLS_SET:
                return jsonify({"error": f"Skill '{skill_name}' sudah ada"}), 409
            execute_db_change("INSERT INTO skills (skill) VALUES (?)", (skill_name,))
            state.VALID_SKILLS_SET.add(skill_lower)
            print(f"✅ Added skill to DB: {skill_name}")
            return jsonify({"status": "success", "message": f"Skill '{skill_name}' ditambahkan"}), 201

        skill_id = request.args.get("id")
        if not skill_id:
            return jsonify({"error": "ID required"}), 400
        rows = fetch_db_rows("SELECT rowid AS id, skill AS name FROM skills WHERE rowid = ?", (int(skill_id),))
        if not rows:
            return jsonify({"error": "Skill not found"}), 404
        skill_name = rows[0]["name"]
        execute_db_change("DELETE FROM skills WHERE rowid = ?", (int(skill_id),))
        state.VALID_SKILLS_SET.discard(skill_name.lower())
        print(f"✅ Deleted skill from DB: {skill_name}")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"❌ Error in manage_skills: {str(e)}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/jobs", methods=["GET", "POST", "DELETE", "OPTIONS"])
def manage_jobs():
    if request.method == "OPTIONS":
        return "", 204
    try:
        if request.method == "GET":
            include_inactive = str(request.args.get("include_inactive", "")).strip().lower() in ("1", "true", "yes", "all")
            status_filter = (request.args.get("status") or "").strip().lower()
            all_data = load_jobs_from_db(active_only=False)
            active_count = sum(1 for job in all_data if job.get("status") != "expired")
            expired_count = len(all_data) - active_count
            visible_jobs = all_data if include_inactive else [job for job in all_data if job.get("status") != "expired"]
            if status_filter in ("active", "expired"):
                visible_jobs = [job for job in visible_jobs if job.get("status") == status_filter]

            jobs_list = []
            for idx, job in enumerate(visible_jobs, 1):
                jobs_list.append(
                    {
                        "id": job.get("id"),
                        "list_id": idx,
                        "title": job.get("job_title", "Unknown"),
                        "company": job.get("company_name", "Unknown"),
                        "url": job.get("job_url", "#"),
                        "postDate": None,
                        "scrapedAt": None,
                        "status": job.get("status", "active"),
                    }
                )

            print(f"✅ Returned {len(jobs_list)} jobs (include_inactive={include_inactive}, active={active_count}, expired={expired_count})")
            return jsonify(
                {
                    "jobs": jobs_list,
                    "active_count": active_count,
                    "expired_count": expired_count,
                    "total_count": len(all_data),
                    "data_last_updated": get_data_last_updated(),
                }
            )

        if request.method == "POST":
            try:
                print("📥 [UPLOAD] Request POST /api/jobs diterima.")
                if "file" not in request.files:
                    print("❌ [UPLOAD] Field 'file' tidak ada di request.")
                    return jsonify({"error": "Tidak ada file yang diunggah (field 'file' kosong)."}), 400

                uploaded = request.files["file"]
                if not uploaded or uploaded.filename == "":
                    print("❌ [UPLOAD] Nama file kosong.")
                    return jsonify({"error": "Nama file kosong."}), 400

                print(f"📄 [UPLOAD] Memproses file: {uploaded.filename}")
                t0 = _time.time()
                try:
                    processed_jobs, err = process_uploaded_jobs_file(uploaded)
                except Exception as e:
                    print(f"❌ [UPLOAD] Error processing uploaded file: {e}")
                    traceback.print_exc()
                    return jsonify({"error": f"Gagal memproses file: {e}"}), 400

                if err:
                    return jsonify({"error": err}), 400
                if not processed_jobs:
                    return jsonify({"error": "Tidak ada baris pekerjaan yang valid di file."}), 400

                parse_elapsed = _time.time() - t0
                print(f"⏱  [UPLOAD] Parsing file selesai dalam {parse_elapsed:.2f}s")
                print(f"📥 [UPLOAD] Memasukkan {len(processed_jobs)} pekerjaan ke database...")

                inserted = 0
                failed = 0
                inserted_jobs = []
                for job in processed_jobs:
                    row_id = insert_job_to_db(job)
                    if row_id is not None:
                        inserted += 1
                        inserted_jobs.append((row_id, job.get("description_text", "")))
                    else:
                        failed += 1

                db_elapsed = _time.time() - t0
                print(f"✅ [UPLOAD] Inserted ke DB: {inserted} sukses, {failed} gagal — total {db_elapsed:.2f}s")

                deleted_duplicates = delete_duplicate_jobs_from_db()
                if deleted_duplicates:
                    print(f"🧹 [UPLOAD] Duplicate jobs removed after import: {deleted_duplicates}")

                try:
                    state.JOB_LIST = load_jobs_from_db()
                    print(f"✅ [UPLOAD] JOB_LIST diperbarui: {len(state.JOB_LIST)} lowongan aktif")
                except Exception as e:
                    print(f"⚠️ Gagal refresh JOB_LIST (data tetap tersimpan): {e}")

                if inserted_jobs:
                    def bg_extract(jobs_snapshot, started_at):
                        print(f"\n🔧 [BG-SKILL] Mulai ekstraksi skill untuk {len(jobs_snapshot)} lowongan...")
                        done = 0
                        for row_id, text in jobs_snapshot:
                            if not text:
                                continue
                            try:
                                skills = extract_skills_manual(text.lower(), state.VALID_SKILLS_SET)
                                if skills:
                                    skills = sanitize_requirement_skill_list(skills)
                                if skills:
                                    execute_db_change("UPDATE jobs SET requirements = ? WHERE rowid = ?", (", ".join(skills), row_id))
                                done += 1
                            except Exception as ex:
                                print(f"⚠️ [BG-SKILL] Error rowid={row_id}: {ex}")
                        elapsed = _time.time() - started_at
                        print(f"✅ [BG-SKILL] Selesai: {done}/{len(jobs_snapshot)} lowongan diperbarui ({elapsed:.1f}s)\n")

                    threading.Thread(target=bg_extract, args=(inserted_jobs, t0), daemon=True).start()
                    print(f"🔧 [UPLOAD] Background skill extraction dimulai ({len(inserted_jobs)} job) — berlangsung di latar belakang")

                total_elapsed = _time.time() - t0
                print(f"✅ [UPLOAD] Upload selesai dalam {total_elapsed:.2f}s: {inserted} ditambahkan, {failed} gagal.")
                print(f"{'=' * 55}\n")
                return jsonify(
                    {
                        "status": "success",
                        "inserted": inserted,
                        "failed": failed,
                        "total_processed": len(processed_jobs),
                        "message": f"{inserted} pekerjaan berhasil ditambahkan ke database.",
                    }
                ), 200
            except Exception as e:
                print(f"❌ [UPLOAD] Unexpected error during POST /api/jobs: {e}")
                traceback.print_exc()
                return jsonify({"error": f"Upload gagal: {str(e)}"}), 500

        job_id = request.args.get("id")
        if job_id:
            try:
                rowid = resolve_job_rowid(job_id)
                if rowid is not None and delete_job_from_db(rowid):
                    state.JOB_LIST = load_jobs_from_db(active_only=True)
                    print(f"✅ Deleted job rowid: {rowid}")
                    return jsonify({"status": "success"}), 200
            except Exception as e:
                print(f"❌ Error deleting job by ID: {e}")
        return jsonify({"error": "Job not found"}), 404
    except Exception as e:
        print(f"❌ Unexpected error in manage_jobs ({request.method}): {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500
