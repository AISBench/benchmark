import json
import os
from pathlib import Path

from datasets import Dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.datasets.utils.video import VideoAsset, image_to_base64
from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_VIDEO_START

TEXT_MAP = {
            2: 'two', 
            3: 'three', 
            4: 'four', 
            5: 'five', 
            6: 'six', 
        }

DEFAULT_NUM_FRAMES = 5 # Default number of frames to sample from each video when loading datasets

@LOAD_DATASET.register_module()
class VideoBenchDataset(BaseDataset):

    @staticmethod
    def load(path:str, video_type:str, num_frames:int=DEFAULT_NUM_FRAMES):
        """
        Load VideoBench dataset from a local directory.

        Parameters
        ----------
        path : str
            Root directory that contains the dataset.
            Inside this directory you MUST have
            1) an `answer/ANSWER.json` file that stores the ground-truth answers,
            2) one or more `*new.json` files that store the questions, choices
            and video metadata.
        video_type : str
            Reserved argument that decides which video field to use.
            Currently only "video_path" is supported, i.e. the local path
            stored in `vid_path`.
        num_frames : int, optional
            Number of frames to sample from each video (not used in this
            implementation, kept for API consistency). Default is 5.

        Returns
        -------
        datasets.Dataset
            A HuggingFace `datasets.Dataset` object where every row is a
            dictionary with the following keys:
            - "video_url"        : str   local path to the video file
            - "video_id"         : str   unique identifier of the video
            - "question"         : str   question text
            - "choices_prompt"   : str   fixed prompt "Choices: " (6 options)
                                        or a single space otherwise
            - "answer"           : str   ground-truth answer string
        """
        path = get_data_path(path, local_mode=True)
        ans_path = path + '/answer/ANSWER.json'
        if not os.path.exists(ans_path):
            raise FileNotFoundError("Cannot find answer file, Please check your datasets!")
        with open(ans_path, 'r', encoding='utf-8') as f_ans:
            answers = json.load(f_ans)
        path = Path(path)
        dataset = []
        for sub_path in path.glob("*new.json"):
            with open(sub_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key in data.keys():
                try:
                    dataset_name = data[key]['vid_path'].split('/')[-2]
                    answer = answers[dataset_name][key]
                    choices = {k: v for k, v in data[key]["choices"].items() if v is not None}
                    choices_prompt = "Choices: " if len(choices)==6 else " "
                    for k in choices:
                        choices_prompt += (k + '.' + choices[k] + ' ')
                    choices_prompt += f'\n Among the {TEXT_MAP.get(len(choices))} options {", ".join(choices.keys())} above,' \
                                        ' the one closest to the correct answer is:'
                    if len(choices) in [2, 3, 5]:
                        choices_prompt += " "

                    # get video_url
                    if video_type == "video_path":
                        video_url = data[key]["vid_path"]
                    elif video_type == "video_base64":
                        base64_frames = []
                        frames = VideoAsset(video_path=data[key]["vid_path"], num_frames=num_frames).pil_images
                        for frame in frames:
                            base64_frame = image_to_base64(frame)
                            base64_frames.append(base64_frame)
                        video_url = ','.join(base64_frames)
                    else:
                        raise ValueError("video_type must be video_path or video_base64")
                    question = data[key]["question"] + choices_prompt
                    content = AIS_VIDEO_START + video_url + AIS_CONTENT_TAG \
                            + AIS_TEXT_START + question + AIS_CONTENT_TAG
                    dataset.append({"video_url": video_url,
                                    "video_id": str(data[key]["video_id"]),
                                    "question": data[key]["question"],
                                    "choices_prompt": choices_prompt,
                                    "content": content,
                                    'answer': answer})
                except:
                    raise ValueError("Please check your datasets!")
                
        return Dataset.from_list(dataset)


class VideoBenchEvaluator(BaseEvaluator):

    def find_choice(self, result):
        choice_list = ['A', 'B', 'C', 'D', 'E', 'F']
        for choice in choice_list:
            if choice in result:
                return choice
        return ""

    def score(self, predictions, references):
        references = [i['answer'] for i in references]
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if self.find_choice(i) == j:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result