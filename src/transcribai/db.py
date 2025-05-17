import os
import sqlite3

DATA_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/database.db")
)


def get_connection():
    conn = sqlite3.connect(DATA_DB_PATH)
    return conn


def init_db():
    os.makedirs(os.path.dirname(DATA_DB_PATH), exist_ok=True)
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            user_telegram_id INTEGER,
            idx INTEGER,
            video_id TEXT,
            video_link TEXT,
            video_file_path TEXT,
            language TEXT,
            PRIMARY KEY(user_telegram_id, idx)
        )
    """
    )
    conn.commit()
    conn.close()


def get_next_index(user_telegram_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT MAX(idx) FROM files WHERE user_telegram_id=?", (user_telegram_id,)
    )
    result = c.fetchone()
    conn.close()
    return (result[0] or 0) + 1


def make_video_id(user_telegram_id, idx):
    return f"{user_telegram_id}_{idx}"


def insert_file(user_telegram_id, idx, video_id, video_link, video_file_path, language):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO files (user_telegram_id, idx, video_id, video_link, video_file_path, language)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (user_telegram_id, idx, video_id, video_link, video_file_path, language),
    )
    conn.commit()
    conn.close()


def get_user_files(user_telegram_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT idx, video_id, video_link, video_file_path, language FROM files WHERE user_telegram_id=? ORDER BY idx",
        (user_telegram_id,),
    )
    files = c.fetchall()
    conn.close()
    return files


def get_file_by_id(user_telegram_id, idx):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT video_id, video_link, video_file_path, language FROM files WHERE user_telegram_id=? AND idx=?",
        (user_telegram_id, idx),
    )
    file = c.fetchone()
    conn.close()
    return file
