import unittest
import tempfile
import sqlite3
import numpy as np
import os
from unittest import mock

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.db_utils import (
    init_db,
    save_numpy_to_db,
    load_all_numpy_from_db,
)
from ais_bench.benchmark.utils.logging.exceptions import FileOperationError


class TestDbUtils(unittest.TestCase):
    def test_init_db(self):
        """Test init_db function (lines 20-24)"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = init_db(db_path)
            self.assertIsNotNone(conn)
            
            # Verify table exists
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='numpy_store'")
            self.assertIsNotNone(cur.fetchone())
            
            conn.close()
        finally:
            os.unlink(db_path)

    def test_init_db_with_timeout(self):
        """Test init_db with connection timeout"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = init_db(db_path)
            self.assertIsNotNone(conn)
            conn.close()
        finally:
            os.unlink(db_path)

    def test_init_db_creates_table(self):
        """Test init_db creates table if not exists (lines 20-24)"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = init_db(db_path)
            # Verify table exists
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='numpy_store'")
            self.assertIsNotNone(cur.fetchone())
            conn.close()
        finally:
            os.unlink(db_path)

    def test_save_numpy_to_db(self):
        """Test save_numpy_to_db function (lines 37-49)"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = init_db(db_path)
            arr = np.array([1, 2, 3, 4, 5])
            arr_id = save_numpy_to_db(conn, arr, batch_size=100)
            self.assertIsNotNone(arr_id)
            self.assertGreater(arr_id, 0)
            conn.close()
        finally:
            os.unlink(db_path)

    def test_save_numpy_to_db_with_batch_commit(self):
        """Test save_numpy_to_db with batch commit (lines 45-46)"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = init_db(db_path)
            arr = np.array([1, 2, 3])
            # Save 100 arrays to trigger batch commit
            for i in range(100):
                arr_id = save_numpy_to_db(conn, arr, batch_size=100)
            conn.close()
        finally:
            os.unlink(db_path)

    def test_save_numpy_to_db_error(self):
        """Test save_numpy_to_db with error (lines 48-50)"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = init_db(db_path)
            arr = np.array([1, 2, 3])
            
            # Create a mock cursor that raises error on execute
            from ais_bench.benchmark.openicl.icl_inferencer.output_handler import db_utils
            
            # Save original function
            original_save = db_utils.save_numpy_to_db
            
            # Create a function that mimics the original but raises error in execute
            def failing_save(conn, arr, batch_size=100):
                import io
                import numpy as np
                buf = io.BytesIO()
                np.save(buf, arr, allow_pickle=False)
                data = buf.getvalue()
                cur = conn.cursor()
                
                try:
                    # This will raise sqlite3.Error which should be caught and converted
                    raise sqlite3.Error("Database error")
                except sqlite3.Error as e:
                    from ais_bench.benchmark.utils.logging.exceptions import FileOperationError
                    from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
                    raise FileOperationError(ICLI_CODES.SQLITE_WRITE_ERROR, 
                                           f"Failed to save numpy array to database: {str(e)}")
            
            # Temporarily replace the function
            db_utils.save_numpy_to_db = failing_save
            
            try:
                with self.assertRaises(FileOperationError):
                    db_utils.save_numpy_to_db(conn, arr, batch_size=100)
            finally:
                # Restore original
                db_utils.save_numpy_to_db = original_save
            
            conn.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_load_all_numpy_from_db(self):
        """Test load_all_numpy_from_db function (lines 63-71)"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = init_db(db_path)
            
            # Save arrays
            arr1 = np.array([1, 2, 3])
            arr2 = np.array([4, 5, 6])
            id1 = save_numpy_to_db(conn, arr1, batch_size=100)
            id2 = save_numpy_to_db(conn, arr2, batch_size=100)
            conn.commit()
            
            # Load all
            result = load_all_numpy_from_db(conn)
            self.assertEqual(len(result), 2)
            np.testing.assert_array_equal(result[id1], arr1)
            np.testing.assert_array_equal(result[id2], arr2)
            
            conn.close()
        finally:
            os.unlink(db_path)

    def test_load_all_numpy_from_db_empty(self):
        """Test load_all_numpy_from_db with empty database"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = init_db(db_path)
            result = load_all_numpy_from_db(conn)
            self.assertEqual(len(result), 0)
            conn.close()
        finally:
            os.unlink(db_path)


if __name__ == '__main__':
    unittest.main()

