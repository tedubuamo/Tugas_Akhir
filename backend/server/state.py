import threading


groq_client = None
ner_pipeline = None
VALID_SKILLS_SET = set()
VALID_PROGRAMS_SET = set()
JOB_LIST = []

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
