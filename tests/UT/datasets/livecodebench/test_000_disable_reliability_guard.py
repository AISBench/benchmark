# Ensure this file name sorts early to run before other livecodebench tests.
# It disables destructive reliability_guard side effects during UT.

def _noop(*args, **kwargs):
    return None

try:
    from ais_bench.benchmark.datasets.livecodebench import testing_util as _tu
    _tu.reliability_guard = _noop
except Exception:
    pass

try:
    from ais_bench.benchmark.datasets.livecodebench import execute_utils as _eu
    _eu.reliability_guard = _noop
except Exception:
    pass

# Provide a trivial test to keep unittest happy
import unittest

class TestDisableReliabilityGuard(unittest.TestCase):
    def test_disabled(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
