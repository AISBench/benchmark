import json
import os
import re
from os import environ
from pathlib import Path
import base64

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_AUDIO_START

from .base import BaseDataset


@LOAD_DATASET.register_module()
class VocalSoundDataset(BaseDataset):

    @staticmethod
    def load(path, audio_type):
        """
        Load a tiny audio classification dataset whose ground-truth label is
        embedded in the file name

        Parameters
        ----------
        path : str
            Directory that contains `.wav` files.  Every file is treated as one
            training / evaluation sample.  The label (answer) is extracted from
            the file name by taking the last underscore-separated substring
            **before** the extension, e.g.
                file_name : 00001_cat.wav  -->  answer = "cat"
                file_name : speaker_42.wav  -->  answer = "42"
        audio_type : str
            How the audio should be returned:
            - "audio_path"   : keep the original file path (str)
            - "audio_base64" : read the whole file and return it as a base64-
            encoded UTF-8 string

        Returns
        -------
        datasets.Dataset
            A HuggingFace `datasets.Dataset` where every row is a dictionary:
                {
                "audio_url": str,  # path or base64 string
                "question" : "To be replaced!",  # static placeholder
                "answer"   : str   # label extracted from file name
                }
        """
        path = get_data_path(path, local_mode=True)
        path = Path(path)
        dataset = []
        for file_path in path.glob("*.wav"):
            try:
                answer = os.path.splitext(file_path)[0].split('_')[-1]
                # get audio_url
                if audio_type == "audio_path":
                    audio_url = str(file_path)
                elif audio_type == "audio_base64":
                    with open(str(file_path), 'rb') as f:
                        data = f.read()
                    audio_url = base64.b64encode(data).decode('utf-8')
                question = "In this audio, what kind of sound can you hear? " + \
                            "A: Laughter, B: Sigh, C: Cough, D: Throat clearing, E: Sneeze, F: Sniff, " + \
                            "Please select the one closest to the correct answer. ASSISTANT:"
                content = AIS_AUDIO_START + audio_url + AIS_CONTENT_TAG \
                            + AIS_TEXT_START + question + AIS_CONTENT_TAG
                dataset.append({"audio_url": audio_url,
                                "content": content,
                                'answer': answer})
            except:
                raise ValueError("Please check your datasets!")
                
        return Dataset.from_list(dataset)

class VocalSoundEvaluator(BaseEvaluator):

    def find_choice(self, result):
        choose_map = {
            "A": "laughter",
            "B": "sigh",
            "C": "cough",
            "D": "throatclearing",
            "E": "sneeze",
            "F": "sniff"
        }
        if result in choose_map.keys():
            return choose_map[result]
        else:
            return ""

    def score(self, predictions, references):
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
            if len(i) > 1:
                i = self.find_choice(i[0])
            count += 1
            if i == j:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result