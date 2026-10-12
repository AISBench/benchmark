import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ais_bench.benchmark.registry import (
    LOAD_DATASET,
    MODELS,
    PARTITIONERS,
    RUNNERS,
    TASKS,
    TEXT_POSTPROCESSORS,
    Registry,
    _submodule_exists,
    build_from_cfg,
    get_locations,
    get_plugin_locations,
    load_class,
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

    def test_get_locations_basic(self):
        """Test get_locations returns basic location."""
        locations = get_locations('test_module')

        self.assertEqual(locations, ['ais_bench.benchmark.test_module'])

    def test_submodule_exists_for_directory_file_and_dotted_path(self):
        """Physical plugin subpackages and modules are detected without import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'models').mkdir()
            (root / 'metrics.py').touch()
            (root / 'openicl' / 'icl_inferencer').mkdir(parents=True)
            pkg = SimpleNamespace(__path__=[tmpdir])

            self.assertTrue(_submodule_exists(pkg, 'models'))
            self.assertTrue(_submodule_exists(pkg, 'metrics'))
            self.assertTrue(
                _submodule_exists(pkg, 'openicl.icl_inferencer'))
            self.assertFalse(_submodule_exists(pkg, 'datasets'))

    def test_submodule_exists_without_package_path(self):
        """Non-package entry-point objects cannot contain plugin submodules."""
        self.assertFalse(_submodule_exists(SimpleNamespace(), 'models'))
        self.assertFalse(
            _submodule_exists(SimpleNamespace(__path__=[]), 'models'))

    @patch('ais_bench.benchmark.registry.entry_points')
    def test_get_plugin_locations_uses_physical_packages_without_import(
            self, mock_entry_points):
        """Discovery filters missing modules and never executes submodule code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid_root = root / 'valid_plugin'
            missing_root = root / 'missing_plugin'
            (valid_root / 'models').mkdir(parents=True)
            missing_root.mkdir()

            valid_entry_point = MagicMock()
            valid_entry_point.load.return_value = SimpleNamespace(
                __name__='valid_plugin', __path__=[str(valid_root)])
            missing_entry_point = MagicMock()
            missing_entry_point.load.return_value = SimpleNamespace(
                __name__='missing_plugin', __path__=[str(missing_root)])
            broken_entry_point = MagicMock()
            broken_entry_point.load.side_effect = RuntimeError('broken plugin')
            mock_entry_points.return_value.select.return_value = [
                valid_entry_point,
                missing_entry_point,
                broken_entry_point,
            ]

            with patch('builtins.__import__') as mock_import:
                locations = get_plugin_locations('models')

        self.assertEqual(locations, ['valid_plugin.models'])
        mock_entry_points.return_value.select.assert_called_once_with(
            group='ais_bench.benchmark_plugins')
        valid_entry_point.load.assert_called_once_with()
        missing_entry_point.load.assert_called_once_with()
        broken_entry_point.load.assert_called_once_with()
        mock_import.assert_not_called()

    @patch('ais_bench.benchmark.registry.entry_points',
           side_effect=RuntimeError('metadata unavailable'))
    def test_get_plugin_locations_handles_discovery_failure(
            self, mock_entry_points):
        """A metadata discovery failure does not break registry creation."""
        self.assertEqual(get_plugin_locations('models'), [])
        mock_entry_points.assert_called_once_with()

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

