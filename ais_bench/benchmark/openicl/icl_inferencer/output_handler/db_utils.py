import sqlite3
import io
import numpy as np

from ais_bench.benchmark.utils.logging.exceptions import FileOperationError
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES

CONNECTION_TIMEOUT = 30

def init_db(db_path: str) -> sqlite3.Connection:
    """
    Initialize database.

    Args:
        db_path (str): Database path

    Returns:
        sqlite3.Connection: Database connection
    """
    conn = sqlite3.connect(db_path, timeout=CONNECTION_TIMEOUT)
    # create table if not exists
    conn.execute("CREATE TABLE IF NOT EXISTS numpy_store (id INTEGER PRIMARY KEY AUTOINCREMENT, arr_blob BLOB NOT NULL)")
    
    return conn

def save_numpy_to_db(conn: sqlite3.Connection, arr: np.ndarray, batch_size: int = 100) -> int:
    """
    Save numpy array to database.

    Args:
        conn (sqlite3.Connection): Database connection
        arr (np.ndarray): Numpy array to save

    Returns:
        int: Last row id
    """
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)   
    data = buf.getvalue()
    cur = conn.cursor()
    
    try:
        cur.execute("INSERT INTO numpy_store (arr_blob) VALUES (?)", (sqlite3.Binary(data),))
        cur_id = cur.lastrowid
        if cur_id % batch_size == 0:
            conn.commit()
        return cur_id
    except sqlite3.Error as e:
        raise FileOperationError(ICLI_CODES.SQLITE_WRITE_ERROR, 
                                 f"Failed to save numpy array to database: {str(e)}",
            )
        
def load_all_numpy_from_db(conn: sqlite3.Connection) -> dict[int, np.ndarray]:
    """
    Load all numpy arrays from the database into a dict {id: np.ndarray}.

    Args:
        conn (sqlite3.Connection): Database connection

    Returns:
        dict[int, np.ndarray]: Mapping from row id to numpy array
    """
    cur = conn.cursor()
    cur.execute("SELECT id, arr_blob FROM numpy_store")

    result = {}
    for row_id, blob in cur.fetchall():
        buf = io.BytesIO(blob)
        arr = np.load(buf, allow_pickle=False)
        result[row_id] = arr
    return result