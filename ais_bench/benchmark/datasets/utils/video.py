import base64
from io import BytesIO
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


def video_to_ndarrays(path: str, num_frames: int = -1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    num_frames = num_frames if num_frames > 0 else total_frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    for idx in range(total_frames):
        ok = cap.grab()  # next img
        if not ok:
            break
        if idx in frame_indices:  # only decompress needed
            ret, frame = cap.retrieve()
            if ret:
                frames.append(frame)

    frames = np.stack(frames)
    if len(frames) < num_frames:
        raise ValueError(f"Could not read enough frames from video file {path}"
                         f" (expected {num_frames} frames, got {len(frames)})")
    return frames


def video_to_pil_images_list(path: str,
                             num_frames: int = -1):
    frames = video_to_ndarrays(path, num_frames)
    return [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in frames
    ]


def image_to_base64(image: Image.Image, fmt='png'):
    output_buffer = BytesIO()
    image.save(output_buffer, format=fmt)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str


@dataclass(frozen=True)
class VideoAsset:
    video_path: str
    num_frames: int = -1

    @property
    def pil_images(self):
        ret = video_to_pil_images_list(self.video_path, self.num_frames)
        return ret

    @property
    def np_ndarrays(self):
        ret = video_to_ndarrays(self.video_path, self.num_frames)
        return ret


