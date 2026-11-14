"""Local SQLite history. No server, no account - the local replacement for Supabase."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at     TEXT    NOT NULL,
  text_excerpt   TEXT    NOT NULL,
  text_hash      TEXT    NOT NULL,
  url            TEXT,
  trust_score    INTEGER NOT NULL,
  band           TEXT    NOT NULL,
  ai_probability REAL,
  toxicity_score REAL,
  fact_verdict   TEXT,
  source_url     TEXT,
  report_json    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def save(self, report: dict[str, Any], url: str | None) -> int:
        row = (
            report["analyzed_at"], report["excerpt"], report["text_hash"], url,
            report["trust_score"], report["band"],
            report["ai_detection"].get("probability"),
            report["toxicity"].get("score"),
            report["fact_check"].get("verdict"),
            report["fact_check"].get("source_url"),
            json.dumps(report),
        )
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO analyses (created_at, text_excerpt, text_hash, url, trust_score,"
                " band, ai_probability, toxicity_score, fact_verdict, source_url, report_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)
            self._conn.commit()
            return int(cur.lastrowid)

    def history(self, limit: int = 50) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, created_at, text_excerpt, trust_score, band, ai_probability,"
                " toxicity_score, fact_verdict, source_url, url FROM analyses"
                " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            total = self._conn.execute("SELECT COUNT(*) AS c FROM analyses").fetchone()["c"]
        return [dict(r) for r in rows], int(total)

    def report(self, analysis_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT report_json FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        return json.loads(row["report_json"]) if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
