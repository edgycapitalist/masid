"""SQLite storage layer for experiment results.

Every trial is recorded with full metadata, agent outputs, and scores.
This provides the raw data for statistical analysis.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT PRIMARY KEY,
    architecture TEXT NOT NULL,
    domain TEXT NOT NULL,
    model TEXT NOT NULL,
    task_id TEXT NOT NULL,
    seed INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Scores (normalized 0-1)
    quality_score REAL,
    correctness REAL,
    completeness REAL,
    coherence REAL,
    integration REAL,

    -- Efficiency
    total_tokens INTEGER,
    total_latency_seconds REAL,
    num_rounds INTEGER,

    -- Coordination
    duplication_rate REAL,
    conflict_rate REAL,
    consistency_score REAL,

    -- Robustness (null if not a fault-injected trial)
    fault_type TEXT,
    fault_agent TEXT,

    -- JSON blobs for detailed data
    agent_scores_json TEXT,
    judge_rationale TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS agent_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    role TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_seconds REAL,
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);

CREATE INDEX IF NOT EXISTS idx_trials_arch ON trials(architecture);
CREATE INDEX IF NOT EXISTS idx_trials_domain ON trials(domain);
CREATE INDEX IF NOT EXISTS idx_trials_model ON trials(model);
CREATE INDEX IF NOT EXISTS idx_outputs_trial ON agent_outputs(trial_id);
"""


class ExperimentDB:
    """SQLite-backed storage for experiment results.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file. Created if it does not exist.
    """

    def __init__(self, db_path: str | Path = "data/experiments.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_trial(
        self,
        trial_id: str,
        architecture: str,
        domain: str,
        model: str,
        task_id: str,
        scores: dict[str, float],
        efficiency: dict[str, Any],
        coordination: dict[str, float],
        agent_scores: dict[str, float],
        judge_rationale: str = "",
        fault_type: Optional[str] = None,
        fault_agent: Optional[str] = None,
        seed: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Save a complete trial record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trials (
                    trial_id, architecture, domain, model, task_id, seed,
                    quality_score, correctness, completeness, coherence, integration,
                    total_tokens, total_latency_seconds, num_rounds,
                    duplication_rate, conflict_rate, consistency_score,
                    fault_type, fault_agent,
                    agent_scores_json, judge_rationale, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial_id, architecture, domain, model, task_id, seed,
                    scores.get("overall", 0.0),
                    scores.get("correctness", 0.0),
                    scores.get("completeness", 0.0),
                    scores.get("coherence", 0.0),
                    scores.get("integration", 0.0),
                    efficiency.get("total_tokens", 0),
                    efficiency.get("total_latency", 0.0),
                    efficiency.get("num_rounds", 0),
                    coordination.get("duplication_rate", 0.0),
                    coordination.get("conflict_rate", 0.0),
                    coordination.get("consistency_score", 0.0),
                    fault_type, fault_agent,
                    json.dumps(agent_scores),
                    judge_rationale,
                    json.dumps(metadata or {}),
                ),
            )

    def save_agent_output(
        self,
        trial_id: str,
        agent_id: str,
        role: str,
        round_number: int,
        content: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_seconds: float = 0.0,
    ) -> None:
        """Save a single agent output."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_outputs
                    (trial_id, agent_id, role, round_number, content,
                     prompt_tokens, completion_tokens, latency_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial_id, agent_id, role, round_number, content,
                    prompt_tokens, completion_tokens, latency_seconds,
                ),
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all_trials(self) -> list[dict]:
        """Return all trial records as dicts."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM trials ORDER BY created_at").fetchall()
            return [dict(r) for r in rows]

    def get_trials_by(
        self,
        architecture: Optional[str] = None,
        domain: Optional[str] = None,
        model: Optional[str] = None,
    ) -> list[dict]:
        """Query trials with optional filters."""
        query = "SELECT * FROM trials WHERE 1=1"
        params: list = []
        if architecture:
            query += " AND architecture = ?"
            params.append(architecture)
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        if model:
            query += " AND model = ?"
            params.append(model)
        query += " ORDER BY created_at"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def count_trials(self) -> int:
        """Return the total number of recorded trials."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as n FROM trials").fetchone()
            return row["n"] if row else 0

    def count_trials_for_cell(
        self,
        architecture: str,
        domain: str,
        task_id: str,
    ) -> int:
        """Count existing trials for a specific arch × domain × task cell."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as n FROM trials "
                "WHERE architecture = ? AND domain = ? AND task_id = ?",
                (architecture, domain, task_id),
            ).fetchone()
            return row["n"] if row else 0

    def max_seed(self) -> int:
        """Return the highest seed used so far, or -1 if empty."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(seed) as m FROM trials"
            ).fetchone()
            val = row["m"] if row else None
            return val if val is not None else -1
