import pytest
import sys

from ais_bench.benchmark.utils.core import valid_global_consts

class DummyLogger:
    def __init__(self):
        self.messages = []
    def warning(self, msg):
        self.messages.append(msg)

@pytest.fixture(autouse=True)
def patch_logger(monkeypatch):
    dummy_logger = DummyLogger()
    monkeypatch.setattr(valid_global_consts, "logger", dummy_logger)
    return dummy_logger

@pytest.fixture(autouse=True)
def patch_global_consts(monkeypatch):
    import types
    dummy = types.SimpleNamespace()
    monkeypatch.setattr(valid_global_consts, "global_consts", dummy)
    return dummy

def test_valid_max_chunk_size_valid(patch_global_consts, patch_logger):
    patch_global_consts.MAX_CHUNK_SIZE = 1024
    assert valid_global_consts.get_max_chunk_size() == 1024
    assert patch_logger.messages == []

def test_valid_max_chunk_size_invalid_type(patch_global_consts, patch_logger):
    patch_global_consts.MAX_CHUNK_SIZE = "bad"
    assert valid_global_consts.get_max_chunk_size() == 65536
    assert "MAX_CHUNK_SIZE is invalid" in patch_logger.messages[0]

def test_valid_max_chunk_size_out_of_range(patch_global_consts, patch_logger):
    patch_global_consts.MAX_CHUNK_SIZE = 0
    assert valid_global_consts.get_max_chunk_size() == 65536
    assert "MAX_CHUNK_SIZE is out of range" in patch_logger.messages[0]

    patch_logger.messages.clear()
    patch_global_consts.MAX_CHUNK_SIZE = 2**25
    assert valid_global_consts.get_max_chunk_size() == 65536
    assert "MAX_CHUNK_SIZE is out of range" in patch_logger.messages[0]

def test_valid_request_time_out_valid(patch_global_consts, patch_logger):
    patch_global_consts.REQUEST_TIME_OUT = 10
    assert valid_global_consts.get_request_time_out() == 10
    assert patch_logger.messages == []

def test_valid_request_time_out_none(patch_global_consts, patch_logger):
    patch_global_consts.REQUEST_TIME_OUT = None
    assert valid_global_consts.get_request_time_out() is None

def test_valid_request_time_out_invalid_type(patch_global_consts, patch_logger):
    patch_global_consts.REQUEST_TIME_OUT = "bad"
    assert valid_global_consts.get_request_time_out() is None
    assert "REQUEST_TIME_OUT is invalid" in patch_logger.messages[0]

def test_valid_request_time_out_out_of_range(patch_global_consts, patch_logger):
    patch_global_consts.REQUEST_TIME_OUT = -1
    assert valid_global_consts.get_request_time_out() is None
    assert "REQUEST_TIME_OUT is out of range" in patch_logger.messages[0]

    patch_logger.messages.clear()
    patch_global_consts.REQUEST_TIME_OUT = 3600 * 25
    assert valid_global_consts.get_request_time_out() is None
    assert "REQUEST_TIME_OUT is out of range" in patch_logger.messages[0]
