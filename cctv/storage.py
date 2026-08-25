"""Event persistence: SQLite database plus file logging."""

import logging
import os
import sqlite3
import threading

from datetime import datetime, timedelta

from config import LOG_DIR, SNAPSHOT_DIR, RETENTION_DAYS


os.makedirs(LOG_DIR, exist_ok=True)


# Set up file logging for security events
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "security.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("SmartCCTV")


# SQLite database that stores all events
DATABASE = os.path.join(LOG_DIR, "events.db")

# A single, long-lived connection reused across events. SQLite connections
# are not safe to share across threads, so every access is serialized under
# a lock. This avoids the repeated open/commit/close churn of the old code.
_connection = None
_db_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    """Return the shared connection, creating it on first use."""
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(DATABASE)
    return _connection


# Create the events table if it does not exist yet
def initialize_database():

    with _db_lock:
        connection = _get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                person TEXT,
                snapshot TEXT
            )
        """)

        connection.commit()


# Save one event to the database and the log file
def log_event(
    event_type,
    person=None,
    snapshot=None
):

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    with _db_lock:
        connection = _get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO events
            (timestamp, event_type, person, snapshot)
            VALUES (?, ?, ?, ?)
            """,
            (
                timestamp,
                event_type,
                person,
                snapshot
            )
        )

        connection.commit()

    logger.info(
        "%s | %s | %s",
        event_type,
        person,
        snapshot
    )


# Delete snapshots, database rows, and log lines older than RETENTION_DAYS
def enforce_retention(days: int = RETENTION_DAYS):

    cutoff = datetime.now() - timedelta(days=days)
    cutoff_iso = cutoff.isoformat(timespec="seconds")
    cutoff_ts = cutoff.timestamp()

    # 1) Snapshot files
    removed = 0
    if os.path.isdir(SNAPSHOT_DIR):
        for fname in os.listdir(SNAPSHOT_DIR):
            path = os.path.join(SNAPSHOT_DIR, fname)
            try:
                if os.path.getmtime(path) < cutoff_ts:
                    os.remove(path)
                    removed += 1
            except OSError:
                pass

    # 2) Database rows
    with _db_lock:
        connection = _get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "DELETE FROM events WHERE timestamp < ?",
            (cutoff_iso,)
        )
        connection.commit()

    # 3) Log file lines (rewrite keeping only recent lines)
    log_path = os.path.join(LOG_DIR, "security.log")
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            kept = [
                line for line in lines
                if _line_is_recent(line, cutoff)
            ]
            if len(kept) != len(lines):
                with open(log_path, "w", encoding="utf-8") as f:
                    f.writelines(kept)
        except OSError:
            pass

    if removed:
        logger.info("Retention: removed %d old snapshot(s)", removed)


def _line_is_recent(line: str, cutoff: datetime) -> bool:
    """True when a log line's leading timestamp is at/after the cutoff."""
    try:
        ts = datetime.fromisoformat(line.split(" | ", 1)[0])
        return ts >= cutoff
    except (ValueError, IndexError):
        return True  # keep unparseable lines
