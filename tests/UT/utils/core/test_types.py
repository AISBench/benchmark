import unittest
from ais_bench.benchmark.utils.core.types import check_meta_json_dict, check_output_config_from_meta_json


class TestCheckMetaJsonDict(unittest.TestCase):
    def test_valid_full_structure(self):
        valid_input = {
            "output_config": {
                "method": "random",
                "params": {
                    "min_value": 1,
                    "max_value": "100",
                    "percentage_distribute": [0.1, 0.2, 0.7]
                }
            },
            "request_count": 100,
            "sampling_mode": "default"
        }
        result = check_meta_json_dict(valid_input)
        self.assertEqual(result, valid_input)

    def test_valid_minimal_structure(self):
        valid_input = {
            "output_config": {
                "method": "fixed",
                "params": {
                    "min_value": "1",
                    "max_value": 100,
                    "percentage_distribute": []
                }
            },
            "request_count": "50",
            "sampling_mode": "random"
        }
        result = check_meta_json_dict(valid_input)
        self.assertEqual(result, valid_input)

    def test_invalid_extra_keys(self):
        invalid_input = {
            "output_config": {
                "method": "random",
                "params": {
                    "min_value": 1,
                    "max_value": 100,
                    "percentage_distribute": [],
                    "extra_key": "value"
                }
            },
            "request_count": 100,
            "sampling_mode": "default",
            "extra_top_key": "value"
        }
        with self.assertRaises(ValueError):
            check_meta_json_dict(invalid_input)

    def test_invalid_type_output_config_method(self):
        invalid_input = {
            "output_config": {
                "method": 123,
                "params": {
                    "min_value": 1,
                    "max_value": 100,
                    "percentage_distribute": []
                }
            },
            "request_count": 100,
            "sampling_mode": "default"
        }
        with self.assertRaises(TypeError):
            check_meta_json_dict(invalid_input)

    def test_invalid_type_request_count(self):
        invalid_input = {
            "output_config": {
                "method": "random",
                "params": {
                    "min_value": 1,
                    "max_value": 100,
                    "percentage_distribute": []
                }
            },
            "request_count": [100],
            "sampling_mode": "default"
        }
        with self.assertRaises(TypeError):
            check_meta_json_dict(invalid_input)

    def test_invalid_type_percentage_distribute(self):
        invalid_input = {
            "output_config": {
                "method": "random",
                "params": {
                    "min_value": 1,
                    "max_value": 100,
                    "percentage_distribute": "not_a_list"
                }
            },
            "request_count": 100,
            "sampling_mode": "default"
        }
        with self.assertRaises(TypeError):
            check_meta_json_dict(invalid_input)


class TestCheckOutputConfigFromMetaJson(unittest.TestCase):
    def test_empty_object(self):
        self.assertFalse(check_output_config_from_meta_json({}))

    def test_missing_output_config(self):
        self.assertFalse(check_output_config_from_meta_json({"other_key": "value"}))

    def test_missing_params(self):
        with self.assertRaises(ValueError):
            check_output_config_from_meta_json({"output_config": {"method": "uniform"}})

    def test_uniform_valid(self):
        config = {
            "output_config": {
                "method": "uniform",
                "params": {
                    "min_value": "10",
                    "max_value": "100"
                }
            }
        }
        self.assertTrue(check_output_config_from_meta_json(config))

    def test_uniform_invalid_values(self):
        config = {
            "output_config": {
                "method": "uniform",
                "params": {
                    "min_value": "f32aasd",
                    "max_value": "234"
                }
            }
        }
        with self.assertRaises(ValueError):
            check_output_config_from_meta_json(config)

    def test_uniform_with_invalid_param(self):
        config = {
            "output_config": {
                "method": "uniform",
                "params": {"x": 121, "z": 123, "aw": 553, "min_value": 12}
            }
        }
        with self.assertRaises(ValueError):
            check_output_config_from_meta_json(config)

    def test_uniform_when_minvalue_bigger_than_maxvalue(self):
        config = {
            "output_config": {
                "method": "uniform",
                "params": {"min_value": 12, "max_value": 10}
            }
        }
        with self.assertRaises(ValueError):
            check_output_config_from_meta_json(config)

    def test_percentage_valid(self):
        config = {
            "output_config": {
                "method": "percentage",
                "params": {
                    "percentage_distribute": [[1000, 0.5], [500, 0.5]]
                }
            }
        }
        self.assertTrue(check_output_config_from_meta_json(config))

    def test_percentage_invalid(self):
        config = {
            "output_config": {
                "method": "percentage",
                "params": {
                    "percentage_distribute": [[1000, 1.5]]
                }
            }
        }
        with self.assertRaises(ValueError):
            check_output_config_from_meta_json(config)

    def test_percentage_missing_distribution(self):
        config = {
            "output_config": {
                "method": "percentage",
                "params": {}
            }
        }
        with self.assertRaises(ValueError):
            check_output_config_from_meta_json(config)

    def test_unsupported_method(self):
        config = {
            "output_config": {
                "method": "invalid_method",
                "params": {"key": "value"}
            }
        }
        with self.assertRaises(ValueError):
            check_output_config_from_meta_json(config)


if __name__ == '__main__':
    unittest.main()