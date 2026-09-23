"""Append-only JSONL logs, one per pipeline stage, so any run can be resumed.

Every record is flushed and fsynced as soon as it is written. On load, a
truncated last line (from a crash mid-write) is skipped.
"""

import json
import os
import time
from pathlib import Path

# Terminal statuses: the case is finished for this stage.
OK = "ok"
REJECTED = "rejected"  # screened out, content-filtered, leaked, etc. Never retried.
# Retryable: counts toward max_attempts_per_case.
ERROR = "error"


class StageLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, list[dict]] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.records.setdefault(rec["case_id"], []).append(rec)

    def append(self, case_id: str, status: str, **fields) -> dict:
        rec = {"case_id": case_id, "status": status, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records.setdefault(case_id, []).append(rec)
        return rec

    def final(self, case_id: str) -> dict | None:
        """The terminal (ok/rejected) record for a case, if any."""
        for rec in reversed(self.records.get(case_id, [])):
            if rec["status"] in (OK, REJECTED):
                return rec
        return None

    def errors(self, case_id: str) -> int:
        return sum(r["status"] == ERROR for r in self.records.get(case_id, []))


def write_jsonl(path: Path, rows) -> None:
    """Write a whole file atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
