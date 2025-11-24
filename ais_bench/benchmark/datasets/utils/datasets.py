import os
import random
import json
import io
import base64
from PIL import Image
import copy

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_IMAGE_START, AIS_AUDIO_START, AIS_VIDEO_START

logger = AISLogger()
# These datasets can only be used to evaluate performance.
ONLY_PERF_DATASETS = [
    "ais_bench.benchmark.datasets.MTBenchDataset",
    "ais_bench.benchmark.datasets.ShareGPTDataset",
    "ais_bench.benchmark.datasets.SyntheticDataset",
]
# Multimodal datasets.
MM_DATASETS = [
    "ais_bench.benchmark.datasets.TEXTVQADataset",
    "ais_bench.benchmark.datasets.VideoBenchDataset",
    "ais_bench.benchmark.datasets.VocalSoundDataset",
]
# Multimodal APIs.
MM_APIS = ["ais_bench.benchmark.models.VLLMCustomAPIChat"]

def get_cache_dir(default_dir):
    # TODO Add any necessary supplementary information for here
    return os.environ.get('AIS_BENCH_DATASETS_CACHE', default_dir)


def get_data_path(dataset_path: str, local_mode: bool = True):
    """return dataset id when getting data from ModelScope/HuggingFace repo, otherwise just
    return local path as is.

    Args:
        dataset_path (str): data path
        local_mode (bool): whether to use local path or
            ModelScope/HuggignFace repo
    """
    # update the path with CACHE_DIR
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../") # site-package
    cache_dir = get_cache_dir(default_dir)

    # For absolute path customized by the users, will not auto download dataset
    if dataset_path.startswith('/'):
        return dataset_path

    # For relative path, with CACHE_DIR
    if local_mode:
        local_path = os.path.join(cache_dir, dataset_path)

        if not os.path.exists(local_path):
            readme_path = os.path.join(default_dir, "README.md")
            raise FileExistsError(f"Dataset path: {local_path} is not exist! " +
                                  "Please check section \"--datasets支持的数据集\" of " +
                                  f"{readme_path} to check how to prepare supported datasets.")
        else:
            return local_path
    else:
        raise TypeError('Customized dataset path type is not a absolute path!')


def get_sample_data(data_list: list, sample_mode: str = "default", request_count: int = 0):
    """Get sample data from data_list.

    Args:
        data_list (list): Data list.
        sample_mode (str): Sample mode.
        request_count (int): Request count.
    
    Raises:
        ValueError: If sample mode is not supported.
        ValueError: If request count is negative.

    Returns:
        list: Sampled data list.
    """
    if not request_count:
        logger.info("If u do not provide 'request_count' when using custom-dataset sampling feature, "
                       "we will sample all available data by default.")
        sample_index = len(data_list)
    elif request_count > len(data_list):
        repeat_times = (request_count // len(data_list)) + (1 if request_count % len(data_list) != 0 else 0)
        data_list = [copy.deepcopy(item) for item in data_list * repeat_times][:request_count]
        sample_index = request_count
    elif request_count < 0:
        raise ValueError("The 'request_count' is negative, we only support positive integer.")
    else:
        sample_index = request_count
    # sampling data
    if sample_mode == "default":
        return [copy.deepcopy(item) for item in data_list[:sample_index]]
    elif sample_mode == "random":
        sampled_items = random.sample(data_list, sample_index)
        return [copy.deepcopy(item) for item in sampled_items]
    elif sample_mode == "shuffle":
        shuffle_data = [copy.deepcopy(item) for item in data_list[:sample_index]]
        random.shuffle(shuffle_data)
        return shuffle_data
    else:
        raise ValueError(f"Sample mode: {sample_mode} is not supported!")
    
def get_meta_json(dataset_path, meta_path):
    ori_meta_path = meta_path
    if not meta_path:
        meta_path = dataset_path + '.meta.json'
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_json_conf = json.load(f)
    else:
        if ori_meta_path:
            # user set meta_path does not exists
            raise ValueError(f'The file path specified by parameter "meta_path" does not exist: {ori_meta_path}')
        meta_json_conf = {}
    return meta_json_conf

def toliststr(s):
    if isinstance(s, str) and len(s) >= 1 and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError

def decode_base64_to_image(base64_string, target_size=-1):
    """Decodes a base64-encoded string into a PIL Image, with optional resizing and mode normalization.

    This function:
      - Decodes the input base64 string into binary image data.
      - Loads it as a PIL `Image` object.
      - Converts images with transparency or palette modes (e.g., 'RGBA', 'P', 'LA') to 'RGB'.
      - Optionally resizes the image to fit within a square of side `target_size` using
        `Image.thumbnail` (preserving aspect ratio).
    Args:
        base64_string (str): A base64-encoded representation of an image file (e.g., PNG, JPEG).
        target_size (int, optional): Maximum width and height for the output image.
            If `target_size > 0`, the image is resized to fit within this bound while
            preserving aspect ratio. Defaults to -1 (no resizing).
    Returns:
        PIL.Image.Image: A normalized RGB image, optionally resized.
    """
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P', 'LA'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    base_dir = os.path.dirname(image_path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    image.save(image_path)

def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip  = re.compile(r'(\d)(,)(\d)')
    periodStrip = re.compile(r'(?<!\d)\.(?!\d)')
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def _process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    articles = ['a', 'an', 'the']
    manualMap = {
        'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    contractions = {
        'aint': "ain't",
        'arent': "aren't",
        'cant': "can't",
        'couldve': "could've",
        'couldnt': "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'dont': "don't",
        'hadnt': "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hed': "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        'hes': "he's",
        'howd': "how'd",
        'howll': "how'll",
        'hows': "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        'Im': "I'm",
        'Ive': "I've",
        'isnt': "isn't",
        'itd': "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        'itll': "it'll",
        "let's": "let's",
        'maam': "ma'am",
        'mightnt': "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        'mightve': "might've",
        'mustnt': "mustn't",
        'mustve': "must've",
        'neednt': "needn't",
        'notve': "not've",
        'oclock': "o'clock",
        'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        'shant': "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        'shouldve': "should've",
        'shouldnt': "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd',
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll",
        'somebodys': "somebody's",
        'someoned': "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        'someonell': "someone'll",
        'someones': "someone's",
        'somethingd': "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        'somethingll': "something'll",
        'thats': "that's",
        'thered': "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        'therere': "there're",
        'theres': "there's",
        'theyd': "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        'theyll': "they'll",
        'theyre': "they're",
        'theyve': "they've",
        'twas': "'twas",
        'wasnt': "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        'weve': "we've",
        'werent': "weren't",
        'whatll': "what'll",
        'whatre': "what're",
        'whats': "what's",
        'whatve': "what've",
        'whens': "when's",
        'whered': "where'd",
        'wheres': "where's",
        'whereve': "where've",
        'whod': "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        'wholl': "who'll",
        'whos': "who's",
        'whove': "who've",
        'whyll': "why'll",
        'whyre': "why're",
        'whys': "why's",
        'wont': "won't",
        'wouldve': "would've",
        'wouldnt': "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        'yall': "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        'youd': "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        'youll': "you'll",
        'youre': "you're",
        'youve': "you've",
    }
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def process_answer(answer):
    answer = answer.replace('\n', ' ')
    answer = answer.replace('\t', ' ')
    answer = answer.strip()
    answer = process_punctuation(answer)
    answer = _process_digit_article(answer)
    return answer

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None
    prediction = str(prediction)
    target = str(target)
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()
    
def anls_compute(groundtruth, prediction):
    gt_answer = ' '.join(groundtruth.strip().lower().split())
    det_answer = ' '.join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt_answer, det_answer)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    values = 0.0 if length == 0 else float(dist) / float(length)
    return values

def process_line(pred, refer, method='vqa_score'):
    ret = {}
    answers = refer
    if isinstance(refer, list):
        answers = refer
    else:
        answers = [refer]
    if method == 'vqa_score':
        answers = [process_answer(x) for x in answers]
        pred = process_answer(pred)
        ret = []
        for current_idx, _ in enumerate(answers):
            otherGTAns = [
                item for ret_gt_idx, item in enumerate(answers)
                if ret_gt_idx != current_idx
            ]
            matchingAns = [
                item for item in otherGTAns if item == pred
            ]
            acc = min(1, float(len(matchingAns)) / 3)
            ret.append(acc)
    elif method == 'anls':
        ret = [anls_compute(x, pred.strip()) for x in answers]
    elif method == 'relaxed_accuracy':
        ret = [relaxed_correctness(x, pred.strip()) for x in answers]
    elif method == 'accuracy':
        ret = [(1.0 if (x.strip().lower() == pred.strip().lower()) else 0.0) for x in answers]
    else:  # default using vqa_score to calculate score
        ret = [x == pred.strip() for x in answers]

    return ret

def get_content_str(msgs):
    content = ""
    for msg in msgs:
        if msg['type']=='text':
            content += AIS_TEXT_START
            content += msg['text']
        elif msg['type']=='image_url':
            content += AIS_IMAGE_START
            content += msg['image_url']
        elif msg['type']=='video_url':
            content += AIS_VIDEO_START
            content += msg['text']
        elif msg['type']=='audio_url':
            content += AIS_AUDIO_START
            content += msg['text']
        content += AIS_CONTENT_TAG
    return content