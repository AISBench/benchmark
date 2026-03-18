import io
import json
import os
import tempfile
import unittest
from base64 import b64decode
from unittest.mock import patch

import pandas as pd
from PIL import Image
from datasets import Dataset

from ais_bench.benchmark.datasets.refcoco.refcoco import (
    RefCOCODataset,
    TEMP_IMAGE_STORE_DIR,
    refcoco_bbox_postprocess,
)
from ais_bench.benchmark.registry import TEXT_POSTPROCESSORS


def _build_test_image_bytes(color=(255, 0, 0), size=(8, 6)):
    image = Image.new('RGB', size, color)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()


class TestRefCOCODataset(unittest.TestCase):
    def test_should_expand_rows_and_convert_bbox_when_loading_valid_refcoco_split(self):
        # given
        image_bytes = _build_test_image_bytes()
        sample_frame = pd.DataFrame([
            {
                'question_id': 7,
                'file_name': 'nested/sample.jpg',
                'image': {'bytes': image_bytes},
                'bbox': [1, 2, 3, 4],
                'answer': ['the cat', 'an orange cone'],
            }
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            shard_path = os.path.join(temp_dir, 'val-00000.parquet')
            with open(shard_path, 'wb'):
                pass

            with patch(
                'ais_bench.benchmark.datasets.refcoco.refcoco.get_data_path',
                return_value=temp_dir,
            ), patch(
                'ais_bench.benchmark.datasets.refcoco.refcoco.pd.read_parquet',
                return_value=sample_frame,
            ):
                # when
                dataset = RefCOCODataset.load('/unused', 'val')

            # then
            self.assertIsInstance(dataset, Dataset)
            self.assertEqual(len(dataset), 2)

            first_row = dataset[0]
            second_row = dataset[1]
            expected_image_path = os.path.join(
                temp_dir,
                TEMP_IMAGE_STORE_DIR,
                'val',
                'nested/sample.jpg',
            )

            self.assertTrue(os.path.exists(expected_image_path))
            self.assertEqual(first_row['image'], expected_image_path)
            self.assertEqual(second_row['image'], expected_image_path)
            self.assertEqual(
                first_row['question'],
                'Locate every object that matches the description "cat" in the image. '
                'Report bbox coordinates in JSON format.',
            )
            self.assertEqual(
                second_row['question'],
                'Locate every object that matches the description "orange cone" in the image. '
                'Report bbox coordinates in JSON format.',
            )

            answer_payload = json.loads(first_row['answer'])
            self.assertEqual(answer_payload['question_id'], 7)
            self.assertEqual(answer_payload['bbox'], [1.0, 2.0, 4.0, 6.0])
            self.assertEqual(answer_payload['image_width'], 8)
            self.assertEqual(answer_payload['image_height'], 6)
            self.assertIn(expected_image_path, first_row['content'])
            self.assertIn('Locate every object', first_row['content'])

    def test_should_return_base64_images_when_loading_refcoco_split_in_base64_mode(self):
        # given
        image_bytes = _build_test_image_bytes()
        sample_frame = pd.DataFrame([
            {
                'question_id': 7,
                'file_name': 'nested/sample.jpg',
                'image': {'bytes': image_bytes},
                'bbox': [1, 2, 3, 4],
                'answer': ['the cat', 'an orange cone'],
            }
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            shard_path = os.path.join(temp_dir, 'val-00000.parquet')
            with open(shard_path, 'wb'):
                pass

            with patch(
                'ais_bench.benchmark.datasets.refcoco.refcoco.get_data_path',
                return_value=temp_dir,
            ), patch(
                'ais_bench.benchmark.datasets.refcoco.refcoco.pd.read_parquet',
                return_value=sample_frame,
            ):
                # when
                dataset = RefCOCODataset.load('/unused', 'val', image_type='image_base64')

            # then
            self.assertIsInstance(dataset, Dataset)
            self.assertEqual(len(dataset), 2)

            first_row = dataset[0]
            expected_image_path = os.path.join(
                temp_dir,
                TEMP_IMAGE_STORE_DIR,
                'val',
                'nested/sample.jpg',
            )

            self.assertEqual(first_row['question'], 'Locate every object that matches the description "cat" in the image. Report bbox coordinates in JSON format.')
            self.assertFalse(os.path.exists(expected_image_path))
            self.assertIsInstance(first_row['image'], str)
            self.assertTrue(first_row['image'])
            self.assertEqual(b64decode(first_row['image'])[:4], b'\xff\xd8\xff\xe0')
            self.assertIn(first_row['image'], first_row['content'])
            self.assertIn('Locate every object', first_row['content'])

            answer_payload = json.loads(first_row['answer'])
            self.assertEqual(answer_payload['bbox'], [1.0, 2.0, 4.0, 6.0])

    def test_should_raise_file_not_found_when_loading_split_without_parquet_shards(self):
        # given
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                'ais_bench.benchmark.datasets.refcoco.refcoco.get_data_path',
                return_value=temp_dir,
            ):
                # when / then
                with self.assertRaises(FileNotFoundError):
                    RefCOCODataset.load('/unused', 'testA')

    def test_should_raise_value_error_when_loading_split_with_invalid_image_payload(self):
        # given
        invalid_frame = pd.DataFrame([
            {
                'question_id': 1,
                'file_name': 'sample.jpg',
                'image': 'invalid-payload',
                'bbox': [0, 0, 1, 1],
                'answer': ['the object'],
            }
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            shard_path = os.path.join(temp_dir, 'val-00000.parquet')
            with open(shard_path, 'wb'):
                pass

            with patch(
                'ais_bench.benchmark.datasets.refcoco.refcoco.get_data_path',
                return_value=temp_dir,
            ), patch(
                'ais_bench.benchmark.datasets.refcoco.refcoco.pd.read_parquet',
                return_value=invalid_frame,
            ):
                # when / then
                with self.assertRaises(ValueError):
                    RefCOCODataset.load('/unused', 'val')

    def test_should_raise_value_error_when_loading_split_with_unsupported_image_type(self):
        with self.assertRaises(ValueError):
            RefCOCODataset.load('/unused', 'val', image_type='inline_bytes')


class TestRefCOCOBBoxPostprocess(unittest.TestCase):
    def test_should_extract_bbox_when_prediction_contains_coordinate_sequence(self):
        # given
        prediction = 'Answer: [1.25, 2, 3.5, 4.75]'

        # when
        bbox = refcoco_bbox_postprocess(prediction)

        # then
        self.assertEqual(bbox, [1.25, 2.0, 3.5, 4.75])

    def test_should_return_default_bbox_when_prediction_has_no_coordinate_sequence(self):
        # given
        prediction = 'No bbox generated'

        # when
        bbox = refcoco_bbox_postprocess(prediction)

        # then
        self.assertEqual(bbox, [0.0, 0.0, 0.0, 0.0])

    def test_should_register_postprocessor_when_refcoco_module_is_imported(self):
        # given / when
        registered_processor = TEXT_POSTPROCESSORS.get('refcoco_bbox_1000')

        # then
        self.assertIs(registered_processor, refcoco_bbox_postprocess)


if __name__ == '__main__':
    unittest.main()
