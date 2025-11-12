import unittest
from unittest.mock import patch, MagicMock
import sys
import importlib

import numpy as np
from PIL import Image

# Import the module using importlib to avoid import path confusion
video_module = importlib.import_module("ais_bench.benchmark.datasets.utils.video")
from ais_bench.benchmark.datasets.utils.video import (
    video_to_ndarrays,
    video_to_pil_images_list,
    image_to_base64,
    VideoAsset,
)


class TestVideoUtils(unittest.TestCase):
    @patch("cv2.VideoCapture")
    def test_video_to_ndarrays_basic(self, mock_capture):
        # Mock a capture object with 5 total frames
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 5  # CAP_PROP_FRAME_COUNT
        # grab() returns True 5 times, then False
        cap.grab.side_effect = [True, True, True, True, True]
        # retrieve returns (ret, frame)
        fake_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        cap.retrieve.return_value = (True, fake_frame)
        mock_capture.return_value = cap

        frames = video_to_ndarrays("/path/to/video.mp4", num_frames=3)
        self.assertEqual(frames.shape[0], 3)
        self.assertEqual(frames.shape[1:], fake_frame.shape)

    @patch("cv2.VideoCapture")
    def test_video_to_ndarrays_not_opened(self, mock_capture):
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_capture.return_value = cap
        with self.assertRaises(ValueError):
            video_to_ndarrays("/bad.mp4", num_frames=1)

    @patch("cv2.VideoCapture")
    def test_video_to_ndarrays_not_enough_frames(self, mock_capture):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 2  # total 2 frames
        cap.grab.side_effect = [True, True]
        fake_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        cap.retrieve.return_value = (True, fake_frame)
        mock_capture.return_value = cap
        with self.assertRaises(ValueError):
            video_to_ndarrays("/path.mp4", num_frames=3)

    @patch.object(video_module, "video_to_ndarrays")
    def test_video_to_pil_images_list(self, mock_ndarrays):
        # 2 frames of 2x2 black
        mock_ndarrays.return_value = np.zeros((2, 2, 2, 3), dtype=np.uint8)
        # Use the function from the module to get the patched version
        images = video_module.video_to_pil_images_list("/path.mp4", num_frames=2)
        self.assertEqual(len(images), 2)
        self.assertTrue(all(isinstance(img, Image.Image) for img in images))

    def test_image_to_base64(self):
        img = Image.new("RGB", (2, 2), color=(255, 0, 0))
        b64 = image_to_base64(img, fmt="PNG")
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 0)

    @patch.object(video_module, "video_to_pil_images_list")
    def test_video_asset_pil_images_property(self, mock_pil_list):
        mock_pil_list.return_value = [Image.new("RGB", (1, 1))]
        asset = VideoAsset("/path.mp4", num_frames=1)
        imgs = asset.pil_images
        self.assertEqual(len(imgs), 1)
        mock_pil_list.assert_called_once()

    @patch.object(video_module, "video_to_ndarrays")
    def test_video_asset_np_ndarrays_property(self, mock_ndarrays):
        mock_ndarrays.return_value = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        asset = VideoAsset("/path.mp4", num_frames=1)
        arr = asset.np_ndarrays
        self.assertEqual(arr.shape[0], 1)
        mock_ndarrays.assert_called_once()


if __name__ == "__main__":
    unittest.main()
