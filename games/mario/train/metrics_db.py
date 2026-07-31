"""SQLite metrics store for Mario training (additive, non-disruptive).

Single writer: the StatsCallback in the main training process. WAL mode lets a
Grafana container read the same DB concurrently without blocking the writer.
Per-episode rows are buffered in memory and flushed on the periodic gate (or
every FLUSH_ROWS), so there is no per-step disk I/O on the hot path.

Every method is defensive; a metrics failure must never crash training — the
caller also guards each call, but connect() swallows its own errors too.
"""
import os
import sqlite3
import time

DB_PATH = os.environ.get("METRICS_DB", "/app/metrics/mario.db")
FLUSH_ROWS = 25   # buffered episode rows before a flush+commit (WAL checkpoints too)

_DDL = """
CREATE TABLE IF NOT EXISTS episodes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT,
    run_name       TEXT,
    ts             REAL,
    timestep       INTEGER,
    env_idx        INTEGER,
    level          TEXT,
    outcome        TEXT,      -- 'clear' | 'death'
    deathless      INTEGER,
    max_x          INTEGER,
    death_x        INTEGER,
    episode_reward REAL,
    status         TEXT,
    coins          INTEGER,
    time_left      INTEGER,
    kills          INTEGER,
    powerups       INTEGER,
    oneups         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_episodes_ts       ON episodes(ts);
CREATE INDEX IF NOT EXISTS idx_episodes_timestep ON episodes(timestep);
CREATE INDEX IF NOT EXISTS idx_episodes_env      ON episodes(env_idx);
CREATE INDEX IF NOT EXISTS idx_episodes_run      ON episodes(run_id);

CREATE TABLE IF NOT EXISTS snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT,
    run_name         TEXT,
    ts               REAL,
    timestep         INTEGER,
    clears_total     INTEGER,
    deaths_total     INTEGER,
    deathless_clears INTEGER,
    episodes_total   INTEGER,
    clear_pct        REAL,
    deathless_rate   REAL,
    deaths_per_clear REAL,
    max_x_reached    INTEGER,
    jump_pct         REAL,
    fps              REAL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts  ON snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_snapshots_run ON snapshots(run_id);

CREATE TABLE IF NOT EXISTS coach_weights (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    run_name        TEXT,
    ts              REAL,
    timestep        INTEGER,
    progress_bonus  REAL,
    velocity_bonus  REAL,
    stuck_penalty   REAL,
    jump_bonus      REAL,
    stuck_threshold INTEGER
);
CREATE INDEX IF NOT EXISTS idx_coach_ts  ON coach_weights(ts);
CREATE INDEX IF NOT EXISTS idx_coach_run ON coach_weights(run_id);
"""

_EPISODE_COLS = (
    "run_id", "run_name", "ts", "timestep", "env_idx", "level", "outcome",
    "deathless", "max_x", "death_x", "episode_reward", "status", "coins",
    "time_left", "kills", "powerups", "oneups",
)
_SNAPSHOT_COLS = (
    "run_id", "run_name", "ts", "timestep", "clears_total", "deaths_total",
    "deathless_clears", "episodes_total", "clear_pct", "deathless_rate",
    "deaths_per_clear", "max_x_reached", "jump_pct", "fps",
)
_COACH_COLS = (
    "run_id", "run_name", "ts", "timestep", "progress_bonus", "velocity_bonus",
    "stuck_penalty", "jump_bonus", "stuck_threshold",
)


def _insert_sql(table, cols):
    return (f"INSERT INTO {table} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})")


def resolve_run_id(resume=False, run_name="", db_path=DB_PATH):
    """Decide the run identity for this process, and return (run_id, run_name).

    Hybrid policy:
      - A run PERSISTS across RESUME=True restarts: the active run_id is stored
        in metrics/run_id.txt next to the DB; on a resume we reuse it so one
        continuous training effort stays a single run.
      - A fresh start (RESUME=False) OR an explicit new RUN_NAME begins a new
        run_id (timestamp-based) and overwrites the marker file.
      - run_name defaults to the run_id when not set in config.
    """
    marker = os.path.join(os.path.dirname(db_path), "run_id.txt")
    run_name = (run_name or "").strip()

    if resume and os.path.exists(marker):
        try:
            with open(marker) as f:
                saved_id, _, saved_name = f.read().strip().partition("\t")
            if saved_id:
                # If the user set a new RUN_NAME on resume, treat it as a new run.
                if not run_name or run_name == saved_name:
                    return saved_id, (saved_name or saved_id)
        except Exception:
            pass

    run_id = time.strftime("run_%Y%m%d_%H%M%S", time.localtime())
    label = run_name or run_id
    try:
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, "w") as f:
            f.write(f"{run_id}\t{label}")
    except Exception:
        pass
    return run_id, label


class MetricsDB:
    def __init__(self, path=DB_PATH):
        self.path = path
        self.conn = None
        self._pending = []  # buffered episode row tuples

    def connect(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout=5000;")
        # Auto-checkpoint the WAL into the main DB file frequently. A read-only
        # reader (Grafana, mounted :ro) sees ONLY the committed main file — it
        # cannot read the -wal — so without regular checkpoints the dashboard
        # reads a near-empty DB. 64 pages keeps the main file current cheaply.
        self.conn.execute("PRAGMA wal_autocheckpoint=64;")
        self.conn.executescript(_DDL)
        self.conn.commit()

    def insert_episode(self, row):
        """Buffer one episode row; flush + commit when the buffer fills so the
        data lands in the main DB file (visible to the read-only Grafana)."""
        self._pending.append(tuple(row.get(c) for c in _EPISODE_COLS))
        if len(self._pending) >= FLUSH_ROWS:
            self._flush()
            if self.conn is not None:
                self.conn.commit()

    def _flush(self):
        if self.conn is None or not self._pending:
            return
        self.conn.executemany(_insert_sql("episodes", _EPISODE_COLS), self._pending)
        self._pending.clear()

    def insert_snapshot(self, row):
        """Insert an aggregate snapshot, flush buffered episodes, and commit.
        This is the natural per-gate commit point (~once every 50k steps)."""
        if self.conn is None:
            return
        self._flush()
        self.conn.execute(_insert_sql("snapshots", _SNAPSHOT_COLS),
                          tuple(row.get(c) for c in _SNAPSHOT_COLS))
        self.conn.commit()

    def insert_coach(self, row):
        if self.conn is None:
            return
        self.conn.execute(_insert_sql("coach_weights", _COACH_COLS),
                          tuple(row.get(c) for c in _COACH_COLS))
        self.conn.commit()

    def close(self):
        if self.conn is None:
            return
        try:
            self._flush()
            self.conn.commit()
        finally:
            self.conn.close()
            self.conn = None
