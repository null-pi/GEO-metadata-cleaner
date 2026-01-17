import sqlite3
import pathlib
from dataclasses import dataclass


@dataclass
class DownloadRecord:
    gse_id: str
    filename: str
    query: str
    status: str = "success"


class GEODatabase:
    def __init__(self, db_path: str = "geo_history.db"):
        current_script_dir = pathlib.Path(__file__).resolve().parent.parent

        self.db_path = current_script_dir / db_path
        self._init_db()

    def _init_db(self):
        """Creates the downloads table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS downloads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gse_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                search_query TEXT,
                download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT
            )
        """
        )
        conn.commit()
        conn.close()

    def add_record(self, record: DownloadRecord):
        """Inserts a new download record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO downloads (gse_id, filename, search_query, status)
            VALUES (?, ?, ?, ?)
        """,
            (record.gse_id, str(record.filename), record.query, record.status),
        )
        conn.commit()
        conn.close()
