import json
import unittest
from typing import Any, cast

from ais_bench.benchmark.openicl.icl_evaluator.bbox_iou_evaluator import (
    BBoxIoUEvaluator,
)
from ais_bench.benchmark.registry import ICL_EVALUATORS


class TestBBoxIoUEvaluator(unittest.TestCase):
    def test_should_score_accuracy_when_prediction_matches_reference_given_normalized_bbox(
        self,
    ):
        # given
        evaluator = BBoxIoUEvaluator(iou_threshold=0.5, coord_scale=1000.0)
        predictions = [[100, 200, 400, 600]]
        references = [
            json.dumps(
                {
                    "bbox": [10, 20, 40, 60],
                    "image_width": 100,
                    "image_height": 100,
                }
            )
        ]

        # when
        result = cast(dict[str, Any], evaluator.score(predictions, references))
        details = cast(list[dict[str, Any]], result["details"])
        detail = details[0]

        # then
        self.assertEqual(result["Accuracy@0.5"], 100.0)
        self.assertEqual(len(result["details"]), 1)
        self.assertTrue(detail["correct"])
        self.assertEqual(detail["pred_bbox_pixel"], [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(detail["coord_mode"], "0-1000")

    def test_should_clip_prediction_to_image_bounds_when_scaling_bbox_given_out_of_range_coordinates(
        self,
    ):
        # given
        evaluator = BBoxIoUEvaluator(clip_to_image=True)

        # when
        scaled_box = evaluator._scale_prediction([-100, 100, 1200, 900], 200, 100)

        # then
        self.assertEqual(scaled_box, [0.0, 10.0, 200.0, 90.0])

    def test_should_report_invalid_detail_when_prediction_cannot_form_valid_bbox_given_reversed_coordinates(
        self,
    ):
        # given
        evaluator = BBoxIoUEvaluator()
        predictions = [[900, 100, 100, 900]]
        references = [
            {
                "bbox": [0, 0, 100, 100],
                "image_width": 100,
                "image_height": 100,
            }
        ]

        # when
        result = cast(dict[str, Any], evaluator.score(predictions, references))
        details = cast(list[dict[str, Any]], result["details"])
        detail = details[0]

        # then
        self.assertEqual(result["Accuracy@0.5"], 0.0)
        self.assertTrue(detail["invalid"])
        self.assertEqual(detail["iou"], 0.0)
        self.assertIsNone(detail["pred_bbox_pixel"])
        self.assertIn("reversed or empty", detail["error"])

    def test_should_return_error_when_prediction_and_reference_lengths_differ_given_mismatched_inputs(
        self,
    ):
        # given
        evaluator = BBoxIoUEvaluator()

        # when
        result = evaluator.score([[0, 0, 10, 10]], [])

        # then
        self.assertIn("error", result)

    def test_should_use_custom_metric_prefix_and_reference_keys_when_scoring_given_custom_schema(
        self,
    ):
        # given
        evaluator = BBoxIoUEvaluator(
            metric_prefix="IoUAccuracy",
            reference_bbox_key="target_bbox",
            image_width_key="width",
            image_height_key="height",
        )
        predictions = [[0, 0, 500, 500]]
        references = [
            {
                "target_bbox": [0, 0, 50, 50],
                "width": 100,
                "height": 100,
            }
        ]

        # when
        result = cast(dict[str, Any], evaluator.score(predictions, references))
        details = cast(list[dict[str, Any]], result["details"])
        detail = details[0]

        # then
        self.assertEqual(result["IoUAccuracy@0.5"], 100.0)
        self.assertTrue(detail["correct"])

    def test_should_register_evaluator_when_bbox_iou_module_is_imported_given_registry_lookup(
        self,
    ):
        # given / when
        registered_evaluator = ICL_EVALUATORS.get("BBoxIoUEvaluator")

        # then
        self.assertIs(registered_evaluator, BBoxIoUEvaluator)

    def test_should_score_accuracy_with_smart_resize_enabled_given_aligned_prediction(
        self,
    ):
        # given
        evaluator = BBoxIoUEvaluator(
            iou_threshold=0.5,
            coord_scale=1000.0,
            smart_resize_cfg={
                "factor": 32,
                "min_pixels": 32 * 32 * 16,
                "max_pixels": 32 * 32 * 4 * 16384,
            },
        )
        # For 100x100 image, smart resize -> 128x128. Build prediction so final scaled bbox matches GT.
        predictions = [[78.125, 156.25, 312.5, 468.75]]
        references = [
            json.dumps(
                {
                    "bbox": [10, 20, 40, 60],
                    "image_width": 100,
                    "image_height": 100,
                }
            )
        ]

        # when
        result = cast(dict[str, Any], evaluator.score(predictions, references))
        details = cast(list[dict[str, Any]], result["details"])
        detail = details[0]

        # then
        self.assertEqual(result["Accuracy@0.5"], 100.0)
        self.assertTrue(detail["correct"])
        self.assertEqual(detail["coord_mode"], "smart_resize_pixels")

    def test_should_not_raise_during_score_when_smart_resize_fails_given_extreme_aspect_ratio(
        self,
    ):
        # given
        evaluator = BBoxIoUEvaluator(
            smart_resize_cfg={
                "factor": 32,
                "min_pixels": 32 * 32 * 16,
                "max_pixels": 32 * 32 * 4 * 16384,
            },
        )
        predictions = [[100, 100, 400, 400]]
        references = [
            {
                "bbox": [0, 0, 1, 10],
                "image_width": 1,
                "image_height": 500,
            }
        ]

        # when
        result = cast(dict[str, Any], evaluator.score(predictions, references))
        details = cast(list[dict[str, Any]], result["details"])
        detail = details[0]

        # then
        self.assertEqual(result["Accuracy@0.5"], 0.0)
        self.assertTrue(detail["invalid"])
        self.assertIn("absolute aspect ratio", detail["error"])


if __name__ == "__main__":
    unittest.main()
