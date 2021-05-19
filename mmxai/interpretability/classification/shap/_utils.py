"""
===============================================================================
utils.py

various utilities for the hateful memes dataset
===============================================================================
"""

import os
import json
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image


class WhitespaceTokenizer(object):
    mask_token_id = 103
    unk_token_id = 100

    def __init__(self) -> None:
        super().__init__()
        self._data = {self.mask_token_id: "[MASK]", self.unk_token_id: "[UNK}"}
        self._inv_data = {v: k for k, v in self._data.items()}
        self._counter = 0

    # def __call__(self, text: str, **kwargs):
    #     return self.encode(text, **kwargs)

    def encode(self, text: str, **kwargs):
        tokens = self.tokenize(text)
        return [self._inv_data[t] for t in tokens]

    def tokenize(self, text: str, **kwargs):
        tokens = text.split(sep=" ")
        # store data if not in
        for t in tokens:
            if t not in self._data.values():
                if (
                    self._counter == self.mask_token_id
                    or self._counter == self.unk_token_id
                ):
                    self._counter += 1
                self._data[self._counter] = t
                self._inv_data[t] = self._counter
                self._counter += 1
        return tokens

    def decode(self, codes: list, **kwargs):
        x = self._data

        def f(c):
            return x[c] if x.get(c) is not None else x[self.unk_token_id]

        return " ".join([f(c) for c in codes])


def read_labels(label_file: str, append_dir: False) -> list:
    """utility that reads data labels

    Args:
        label_file: path to jsonl file that contains the metadata
        append_dir: whether to append the directory path to 'img' element of each line

    Returns:
        list of dictionaries, each element is one json line

    """
    dir_name = os.path.dirname(label_file) + "/" if append_dir == True else None
    with open(label_file, "r") as f:
        json_lines = f.read().splitlines()
    labels = []
    for js in json_lines:
        dic = json.loads(js)
        if dir_name is not None:
            dic["img"] = dir_name + dic["img"]
        labels.append(dic)
    return labels


def parse_labels(labels: List[Dict], img_to_array=False, separate_outputs=False):
    """loads whats in the labels into a list of tuples (for now)
    WARNING: VERY MEMORY INTENSIVE FOR MANY IMAGES

    Args:
        labels: a list of dictionary where each element is one json line that has
            'img': str, path to the image
            'text': str, caption to the image
            size: N
        img_to_array: whether to change image to its numpy array representation
            if False will be an PIL image object
        saparate_outputs: whether to separate images and texts

    Returns:
        list of tuples, each contain (img_i, caption_i); img_i could be numpy array
        or PIL object, caption_i will be string

    """
    out = []
    images = []
    texts = []
    for i, lb in enumerate(labels):
        img = Image.open(lb["img"])
        if img_to_array == True:
            img = np.array(img, dtype=np.uint8)
            if len(img.shape) == 2:
                img = img[..., np.newaxis]
        txt = lb["text"]
        if separate_outputs == True:
            texts.append(txt)
            images.append(img)
        else:
            out.append((img, txt))
    if separate_outputs == True:
        # in case the images are of different shapes
        images = (
            np.array(images)
            if all([i.shape == images[0].shape for i in images])
            else np.array(images, dtype=object)
        )
        return images, np.array(texts)
    else:
        return np.array(out)


def arr_to_img(images):
    # greyscale to rgb
    if len(images[0].shape) == 3 and images[0].shape[2] == 1:
        images = [
            Image.fromarray(img.astype(np.uint8).squeeze(2), "L") for img in images
        ]
    else:
        images = [Image.fromarray(img.astype(np.uint8)) for img in images]

    return images
