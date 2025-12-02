import unittest
from unittest.mock import patch, MagicMock
import importlib

from ais_bench.benchmark.registry import (
    load_class,
    get_locations,
    Registry,
    PARTITIONERS,
    RUNNERS,
    TASKS,
    MODELS,
    LOAD_DATASET,
    TEXT_POSTPROCESSORS,
    build_from_cfg
)


class TestRegistry(unittest.TestCase):
    """Tests for registry module."""

    def test_load_class_success(self):
        """Test load_class with valid class path."""
        result = load_class('unittest.TestCase')
        self.assertEqual(result, unittest.TestCase)

    def test_load_class_invalid_module(self):
        """Test load_class with invalid module."""
        with self.assertRaises(ValueError) as cm:
            load_class('nonexistent.module.Class')
        self.assertIn("无法加载类", str(cm.exception))

    def test_load_class_invalid_class(self):
        """Test load_class with invalid class name."""
        with self.assertRaises(ValueError):
            load_class('unittest.NonexistentClass')

    @patch('ais_bench.benchmark.registry.entry_points')
    def test_get_locations_basic(self, mock_entry_points):
        """Test get_locations returns basic location."""
        mock_entry_points.return_value.select.return_value = []

        locations = get_locations('test_module')

        self.assertIn('ais_bench.benchmark.test_module', locations)

    @patch('ais_bench.benchmark.registry.entry_points')
    @patch('builtins.__import__')
    def test_get_locations_with_plugin(self, mock_import, mock_entry_points):
        """Test get_locations with plugin entry point."""
        mock_entry_point = MagicMock()
        mock_pkg = MagicMock()
        mock_pkg.__name__ = 'plugin.package'
        mock_entry_point.load.return_value = mock_pkg
        mock_entry_points.return_value.select.return_value = [mock_entry_point]
        mock_import.return_value = MagicMock()

        locations = get_locations('test_module')

        # Should have at least the base location, and possibly plugin location
        self.assertGreaterEqual(len(locations), 1)
        self.assertIn('ais_bench.benchmark.test_module', locations)
        # If import succeeds, plugin location should be added
        if len(locations) > 1:
            self.assertIn('plugin.package.test_module', locations)

    def test_registry_register_module(self):
        """Test Registry register_module method."""
        registry = Registry('test_registry')

        @registry.register_module('test_name')
        class TestClass:
            pass

        self.assertIn('test_name', registry)

    def test_registry_register_module_force(self):
        """Test Registry register_module with force=True."""
        registry = Registry('test_registry')

        @registry.register_module('test_name')
        class TestClass1:
            pass

        # Should allow re-registration with force=True (default)
        @registry.register_module('test_name', force=True)
        class TestClass2:
            pass

        self.assertIn('test_name', registry)

    def test_build_from_cfg(self):
        """Test build_from_cfg function."""
        # Register a test module
        @PARTITIONERS.register_module('test_partitioner')
        class TestPartitioner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        cfg = {'type': 'test_partitioner', 'param': 'value'}
        result = build_from_cfg(cfg)

        self.assertIsInstance(result, TestPartitioner)
        self.assertEqual(result.kwargs['param'], 'value')

    def test_registry_instances_exist(self):
        """Test that all registry instances are created."""
        self.assertIsInstance(PARTITIONERS, Registry)
        self.assertIsInstance(RUNNERS, Registry)
        self.assertIsInstance(TASKS, Registry)
        self.assertIsInstance(MODELS, Registry)
        self.assertIsInstance(LOAD_DATASET, Registry)
        self.assertIsInstance(TEXT_POSTPROCESSORS, Registry)


if __name__ == "__main__":
    unittest.main()

