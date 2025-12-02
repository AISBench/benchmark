import json
import base64

from datasets import Dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.file import check_mm_custom
from ais_bench.benchmark.datasets.utils.video import VideoAsset, image_to_base64
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_IMAGE_START, AIS_VIDEO_START, AIS_AUDIO_START

from .base import BaseDataset

logger = AISLogger()

DATA_TYPE_TAG_DICT = {
    "image": AIS_IMAGE_START,
    "video": AIS_VIDEO_START,
    "audio": AIS_AUDIO_START
}

@LOAD_DATASET.register_module()
class MMCustomDataset(BaseDataset):

    @staticmethod
    def load(path, mm_type, num_frames=5):
        """
        Load a MM-style dataset that pairs images with question-answer
        annotations.

        Parameters
        ----------
        path : str
            Full path to the **question file** (usually `*_questions.json` or
            similar).  The corresponding ground-truth answer file is expected to
            live in the same directory and to have the suffix
            `_annotations.json`, e.g.
                /foo/bar/train_questions.json   # <-- `path`
                /foo/bar/train_annotations.json # <-- auto-detected
        mm_type : str
            How the image should be returned:
            - "path"   : keep the original file path (str)
            - "base64" : read the file and return it as a base64-encoded
            string (str)
        """
        path = get_data_path(path, local_mode=True)
        if not check_mm_custom(path):
            raise AISBenchConfigError(UTILS_CODES.MM_CUSTOM_DATASET_WRONG_FORMAT,
                f"Invalid dataset: {path} , please check whether the dataset is a MM-style dataset!"
            )

        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                content = ""
                line = json.loads(line.strip())
                data_type = line["type"]
                if mm_type == "path":
                    for path in line["path"]:
                        content += DATA_TYPE_TAG_DICT[data_type]
                        content += path
                        content += AIS_CONTENT_TAG
                elif mm_type == "base64":
                    if data_type in ["image", "audio"]:
                        for path in line['path']:
                            with open(path, 'rb') as f:
                                binary_data = f.read()
                            data_base64 = base64.b64encode(binary_data).decode("utf-8")
                            content += DATA_TYPE_TAG_DICT[data_type]
                            content += data_base64
                            content += AIS_CONTENT_TAG
                    elif data_type == "video":
                        for path in line['path']:
                            base64_frames = []
                            frames = VideoAsset(video_path=path, num_frames=num_frames).pil_images
                            for frame in frames:
                                base64_frame = image_to_base64(frame)
                                base64_frames.append(base64_frame)
                            data_base64 = ','.join(base64_frames)
                            content += DATA_TYPE_TAG_DICT[data_type]
                            content += data_base64
                            content += AIS_CONTENT_TAG
                else:
                    raise AISBenchConfigError(
                        UTILS_CODES.MM_CUSTOM_DATASET_WRONG_FORMAT,
                        f"Param 'mm_type' does not match the data type of dataset: {path} , please check it!"
                    )
                content += AIS_TEXT_START
                content += line['question']
                line['content'] = content
                dataset.append(line)
        return Dataset.from_list(dataset)


class MMCustomEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        result = {'accuracy': 1}
        return result