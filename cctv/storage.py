"""Event persistence: SQLite database plus file logging."""

import logging
import os
import sqlite3

from datetime import datetime

from config import LOG_DIR


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


# Create the events table if it does not exist yet
def initialize_database():

    connection = sqlite3.connect(DATABASE)

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
    connection.close()


# Save one event to the database and the log file
def log_event(
    event_type,
    person=None,
    snapshot=None
):

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    connection = sqlite3.connect(DATABASE)

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
    connection.close()

    logger.info(
        "%s | %s | %s",
        event_type,
        person,
        snapshot
    )
