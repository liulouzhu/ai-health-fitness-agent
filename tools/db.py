"""PostgreSQL connection management for food database.

Reuses DATABASE_URL from environment - the same DB used by LangGraph checkpointer,
but uses a separate schema / tables to avoid conflicts.
"""

import os
import psycopg
from contextlib import contextmanager

# Load .env if exists (same pattern used by other modules)
from dotenv import load_dotenv
load_dotenv()


@contextmanager
def get_connection():
    """Get a PostgreSQL connection.

    Yields:
        psycopg.Connection: Database connection. Caller must close it.
    """
    # Lazy check: re-read env each call in case it changed
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    conn = psycopg.connect(db_url)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_cursor():
    """Get a PostgreSQL cursor with auto-commit on close.

    Yields:
        psycopg.Cursor: Database cursor.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


def ensure_tables():
    """Create food nutrition tables if they don't exist.

    Uses IF NOT EXISTS so it's safe to call multiple times.
    """
    with get_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS foods (
                id SERIAL PRIMARY KEY,
                name VARCHAR(200) NOT NULL UNIQUE,
                category VARCHAR(100),
                description TEXT,
                source VARCHAR(200),
                calories_per_100g FLOAT,
                protein_per_100g FLOAT,
                fat_per_100g FLOAT,
                carbs_per_100g FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS food_aliases (
                id SERIAL PRIMARY KEY,
                food_id INTEGER NOT NULL REFERENCES foods(id) ON DELETE CASCADE,
                alias VARCHAR(200) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(food_id, alias)
            );

            CREATE TABLE IF NOT EXISTS food_portions (
                id SERIAL PRIMARY KEY,
                food_id INTEGER NOT NULL REFERENCES foods(id) ON DELETE CASCADE,
                portion_name VARCHAR(200) NOT NULL,
                portion_grams FLOAT NOT NULL,
                calories_per_portion FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(food_id, portion_name)
            );

            CREATE INDEX IF NOT EXISTS idx_food_aliases_alias
                ON food_aliases USING gin(to_tsvector('simple', alias));

            CREATE INDEX IF NOT EXISTS idx_foods_name
                ON foods(name);
        """)

        print("[DB] Tables ensured: foods, food_aliases, food_portions")