from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Dict, List, Union

from mmengine.config import ConfigDict


def safe_format(input_str: str, **kwargs) -> str:
    """Safely formats a string with the given keyword arguments. If a keyword
    is not found in the string, it will be ignored.

    Args:
        input_str (str): The string to be formatted.
        **kwargs: The keyword arguments to be used for formatting.

    Returns:
        str: The formatted string.
    """
    # import re
    # segs = [input_str]
    # for k, v in kwargs.items():
    #     regex = re.compile(f'(?<={{{k}}})(?={{{k}}})|({{{k}}})')
    #     segs = [regex.split(seg) for seg in segs]
    #     segs = sum(segs, [])
    # replace_dict = {f'{{{k}}}': str(v) for k, v in kwargs.items()}
    # segs = [replace_dict.get(seg, seg) for seg in segs]
    # output_str = ''.join(segs)
    # return output_str

    for k, v in kwargs.items():
        input_str = input_str.replace(f'{{{k}}}', str(v))
    return input_str


def get_prompt_hash(dataset_cfg: Union[ConfigDict, List[ConfigDict]]) -> str:
    """Get the hash of the prompt configuration.

    Args:
        dataset_cfg (ConfigDict or list[ConfigDict]): The dataset
            configuration.

    Returns:
        str: The hash of the prompt configuration.
    """
    if isinstance(dataset_cfg, list):
        if len(dataset_cfg) == 1:
            dataset_cfg = dataset_cfg[0]
        else:
            hashes = ','.join([get_prompt_hash(cfg) for cfg in dataset_cfg])
            hash_object = hashlib.sha256(hashes.encode())
            return hash_object.hexdigest()
    if 'reader_cfg' in dataset_cfg.infer_cfg:
        # new config
        reader_cfg = dict(type='DatasetReader',
                          input_columns=dataset_cfg.reader_cfg.input_columns,
                          output_column=dataset_cfg.reader_cfg.output_column)
        dataset_cfg.infer_cfg.reader = reader_cfg
        if 'train_split' in dataset_cfg.infer_cfg.reader_cfg:
            dataset_cfg.infer_cfg.retriever[
                'index_split'] = dataset_cfg.infer_cfg['reader_cfg'][
                    'train_split']
        if 'test_split' in dataset_cfg.infer_cfg.reader_cfg:
            dataset_cfg.infer_cfg.retriever[
                'test_split'] = dataset_cfg.infer_cfg.reader_cfg.test_split
        for k, v in dataset_cfg.infer_cfg.items():
            dataset_cfg.infer_cfg[k]['type'] = v['type'].split('.')[-1]
    # A compromise for the hash consistency
    if 'fix_id_list' in dataset_cfg.infer_cfg.retriever:
        fix_id_list = dataset_cfg.infer_cfg.retriever.pop('fix_id_list')
        dataset_cfg.infer_cfg.inferencer['fix_id_list'] = fix_id_list
    d_json = json.dumps(dataset_cfg.infer_cfg.to_dict(), sort_keys=True)
    hash_object = hashlib.sha256(d_json.encode())
    return hash_object.hexdigest()


def is_mm_prompt(prompt):
    # input type1: custom language (aime2024_gen)
    # input type2: chat language (aime2024_gen_0_shot_chat_prompt)
    # input type3: multi-modal input (textvqa_gen)
    # Return: Only return True when input type3
    if not isinstance(prompt, list):
        return False
    for message in prompt:
        content = message.get('content')
        if not isinstance(content, list):
            return False
        for mm_data in content:
            if mm_data.get('type') in ('text','image_url', 'video_url', 'audio_url'):
                return True
    return False


class PromptList(list):
    """An enhanced list, used for intermidate representation of a prompt."""

    def format(self, **kwargs) -> PromptList:
        """Replaces all instances of 'src' in the PromptList with 'dst'.

        Args:
            src (str): The string to be replaced.
            dst (PromptType): The string or PromptList to replace with.

        Returns:
            PromptList: A new PromptList with 'src' replaced by 'dst'.

        Raises:
            TypeError: If 'dst' is a PromptList and 'src' is in a dictionary's
            'prompt' key.
        """
        new_list = PromptList()
        for item in self:
            if isinstance(item, Dict):
                new_item = deepcopy(item)
                if 'prompt' in item:
                    new_item['prompt'] = safe_format(item['prompt'], **kwargs)
                elif 'prompt_mm' in item:
                    self.format_mm(new_item=new_item, item=item, **kwargs)
                new_list.append(new_item)
            else:
                new_list.append(safe_format(item, **kwargs))
        return new_list

    def format_mm(self, new_item, item, **kwargs) -> PromptList:
        """
        Format multimedia fields (image/video/audio/text) in-place.
        If the field value is a dict containing {'url': ...}, only the url
        part is formatted; otherwise the whole value is formatted.
        """
        mm = item.get('prompt_mm', {})
        new_mm = new_item.setdefault('prompt_mm', {})

        #image_url/video_url/audio_url
        for field in ('image_url', 'video_url', 'audio_url'):
            if field not in mm:
                continue
            val = mm[field]
            if isinstance(val, dict) and val.get('url') is not None:
                val = val.copy()
                val['url'] = safe_format(val['url'], **kwargs)
            else:
                val = safe_format(val, **kwargs)
            new_mm[field] = val

        # text
        if 'text' in mm:
            new_mm['text'] = safe_format(mm['text'], **kwargs)

    def replace(self, src: str, dst: Union[str, PromptList]) -> PromptList:
        """Replaces all instances of 'src' in the PromptList with 'dst'.

        Args:
            src (str): The string to be replaced.
            dst (PromptType): The string or PromptList to replace with.

        Returns:
            PromptList: A new PromptList with 'src' replaced by 'dst'.

        Raises:
            TypeError: If 'dst' is a PromptList and 'src' is in a dictionary's
            'prompt' key.
        """
        new_list = PromptList()
        for item in self:
            if isinstance(item, str):
                if isinstance(dst, str):
                    new_list.append(item.replace(src, dst))
                elif isinstance(dst, PromptList):
                    split_str = item.split(src)
                    for i, split_item in enumerate(split_str):
                        if split_item:
                            new_list.append(split_item)
                        if i < len(split_str) - 1:
                            new_list += dst
            elif isinstance(item, Dict):
                new_item = deepcopy(item)
                if 'prompt' in item:
                    if src in item['prompt']:
                        if isinstance(dst, PromptList):
                            raise TypeError(
                                f'Found keyword {src} in a dictionary\'s '
                                'prompt key. Cannot replace with a '
                                'PromptList.')
                        new_item['prompt'] = new_item['prompt'].replace(
                            src, dst)
                new_list.append(new_item)
            else:
                new_list.append(item.replace(src, dst))
        return new_list

    def __add__(self, other: Union[str, PromptList]) -> PromptList:
        """Adds a string or another PromptList to this PromptList.

        Args:
            other (PromptType): The string or PromptList to be added.

        Returns:
            PromptList: A new PromptList that is the result of the addition.
        """
        if not other:
            return PromptList([*self])
        if isinstance(other, str):
            return PromptList(self + [other])
        else:
            return PromptList(super().__add__(other))

    def __radd__(self, other: Union[str, PromptList]) -> PromptList:
        """Implements addition when the PromptList is on the right side of the
        '+' operator.

        Args:
            other (PromptType): The string or PromptList to be added.

        Returns:
            PromptList: A new PromptList that is the result of the addition.
        """
        if not other:
            return PromptList([*self])
        if isinstance(other, str):
            return PromptList([other, *self])
        else:
            return PromptList(other + self)

    def __iadd__(self, other: Union[str, PromptList]) -> PromptList:
        """Implements in-place addition for the PromptList.

        Args:
            other (PromptType): The string or PromptList to be added.

        Returns:
            PromptList: The updated PromptList.
        """
        if not other:
            return self
        if isinstance(other, str):
            self.append(other)
        else:
            super().__iadd__(other)
        return self

    def __str__(self) -> str:
        """Converts the PromptList into a string.

        Returns:
            str: The string representation of the PromptList.

        Raises:
            TypeError: If there's an item in the PromptList that is not a
            string or dictionary.
        """
        res = []
        for item in self:
            if isinstance(item, str):
                res.append(item)
            elif isinstance(item, dict):
                if 'prompt' in item:
                    res.append(json.dumps(item['prompt']))
            else:
                raise TypeError("Invalid type in prompt list when "
                                "converting to string")
        return "".join(res)
