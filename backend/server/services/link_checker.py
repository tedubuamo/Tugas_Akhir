import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import requests

from .. import state
from .data_access import load_jobs_from_db, update_job_active_state


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
    return any(phrase in text_lower for phrase in expired_phrases)


def check_url_status(url, timeout=5):
    if not url or url == "#":
        return {"url": url, "status": "error", "status_code": None, "reason": "Invalid URL"}

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            response = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
        except Exception:
            try:
                get_response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
                if 200 <= get_response.status_code < 300:
                    if page_text_indicates_expired(get_response.text):
                        return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": "Page indicates job is no longer advertised"}
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "OK (GET fallback)"}
                if 300 <= get_response.status_code < 400:
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "Redirect (OK, GET fallback)"}
                return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": f"HTTP {get_response.status_code} on GET fallback"}
            except Exception as e2:
                return {"url": url, "status": "expired", "status_code": None, "reason": f"HEAD+GET error: {str(e2)}"}

        status_code = response.status_code
        if status_code >= 500:
            return {"url": url, "status": "expired", "status_code": status_code, "reason": f"HTTP {status_code}"}

        if 200 <= status_code < 400:
            try:
                get_response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
                if 200 <= get_response.status_code < 300:
                    if page_text_indicates_expired(get_response.text):
                        return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": "Page indicates job is no longer advertised"}
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "OK"}
                if 300 <= get_response.status_code < 400:
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "Redirect (OK)"}
                return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": f"HTTP {get_response.status_code} on GET request"}
            except Exception as e:
                return {"url": url, "status": "expired", "status_code": None, "reason": f"GET error: {str(e)}"}

        if status_code in (400, 401, 403, 405):
            try:
                get_response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
                if 200 <= get_response.status_code < 300:
                    if page_text_indicates_expired(get_response.text):
                        return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": "Page indicates job is no longer advertised"}
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "OK (GET confirmed)"}
                if 300 <= get_response.status_code < 400:
                    return {"url": url, "status": "active", "status_code": get_response.status_code, "reason": "Redirect (OK, GET confirmed)"}
                return {"url": url, "status": "expired", "status_code": get_response.status_code, "reason": f"HTTP {get_response.status_code} on GET after HEAD {status_code}"}
            except Exception as e:
                return {"url": url, "status": "expired", "status_code": None, "reason": f"HEAD {status_code} then GET error: {str(e)}"}

        return {"url": url, "status": "expired", "status_code": status_code, "reason": f"HTTP {status_code}"}
    except requests.exceptions.Timeout:
        return {"url": url, "status": "expired", "status_code": None, "reason": "Timeout - Server tidak merespon"}
    except requests.exceptions.ConnectionError:
        return {"url": url, "status": "expired", "status_code": None, "reason": "Connection error - Tidak dapat terhubung"}
    except Exception as e:
        return {"url": url, "status": "expired", "status_code": None, "reason": f"Error: {str(e)}"}


def _run_link_check():
    try:
        jobs_data = load_jobs_from_db()
        active_jobs = [job for job in jobs_data if job.get("status") == "active"]
        total = len(active_jobs)
        print(f"\n🔍 [BG] Mulai cek URL untuk {total} lowongan aktif...")

        with state.CHECK_LOCK:
            state.CHECK_STATE.update(total=total, checked=0, active=0, expired=0, errors=0, results=[])

        results = []
        active_count = expired_count = error_count = 0
        expired_rowids = []

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(check_url_status, job.get("job_url", "#"), 4): idx for idx, job in enumerate(active_jobs)}
            for future in as_completed(futures):
                idx = futures[future]
                job = active_jobs[idx]
                try:
                    check_result = future.result()
                    status = check_result.get("status", "error")
                    if status == "active":
                        job["status"] = "active"
                        active_count += 1
                    else:
                        job["status"] = "expired"
                        if update_job_active_state(job.get("id"), False):
                            expired_rowids.append(job.get("id"))
                        expired_count += 1
                    results.append(
                        {
                            "id": job.get("id"),
                            "title": job.get("job_title", "Unknown"),
                            "url": check_result.get("url"),
                            "status": job["status"],
                            "status_code": check_result.get("status_code"),
                            "reason": check_result.get("reason", ""),
                        }
                    )
                except Exception as e:
                    print(f"❌ [BG] Error cek job {idx + 1}: {e}")
                    error_count += 1
                    results.append(
                        {
                            "id": job.get("id"),
                            "title": job.get("job_title", "Unknown"),
                            "url": job.get("job_url", "#"),
                            "status": "error",
                            "reason": str(e),
                        }
                    )

                with state.CHECK_LOCK:
                    state.CHECK_STATE.update(checked=len(results), active=active_count, expired=expired_count, errors=error_count)

        if expired_rowids:
            try:
                state.JOB_LIST = load_jobs_from_db()
            except Exception as e:
                print(f"⚠️ [BG] Gagal refresh JOB_LIST: {e}")

        with state.CHECK_LOCK:
            state.CHECK_STATE.update(
                running=False,
                done=True,
                results=results,
                active=active_count,
                expired=expired_count,
                errors=error_count,
                checked=len(results),
                finished_at=datetime.datetime.now().isoformat(),
            )
        print(f"✅ [BG] Selesai: {active_count} aktif, {expired_count} expired, {error_count} error.")
    except Exception as e:
        print(f"❌ [BG] Error fatal di _run_link_check: {e}")
        with state.CHECK_LOCK:
            state.CHECK_STATE.update(running=False, done=True, error=str(e), finished_at=datetime.datetime.now().isoformat())


def start_link_check():
    with state.CHECK_LOCK:
        if state.CHECK_STATE.get("running"):
            return False
        state.CHECK_STATE.update(
            running=True,
            done=False,
            total=0,
            checked=0,
            active=0,
            expired=0,
            errors=0,
            results=[],
            error=None,
            started_at=datetime.datetime.now().isoformat(),
            finished_at=None,
        )
    thread = threading.Thread(target=_run_link_check, daemon=True)
    thread.start()
    return True


def get_link_check_snapshot():
    with state.CHECK_LOCK:
        return dict(state.CHECK_STATE)
