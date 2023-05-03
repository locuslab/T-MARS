from functools import partial

import torchvision.transforms as T
from dataset2metadata.augmentations import SquarePadResizeNorm
from clip import clip
import simdjson
from typing import List

CLIP_SIZE = 224
FACES_SIZE = 224
DEDUP_SIZE = 512
TEXT_DETECTION_SIZE=768

CLIP_IMAGE_TRANFORM = clip._transform(n_px=CLIP_SIZE)
CLIP_TEXT_TOKENIZER = partial(clip.tokenize, truncate=True)
FACES_IMAGE_TRANFORM = SquarePadResizeNorm(img_size=FACES_SIZE)

TEXT_DETECTION_IMAGE_TRANFORM = T.Compose([
    T.ToTensor(),
    # convert to numpy array
    T.Lambda(lambda x: x.numpy()),
])

DEDUP_IMAGE_TRANFORM = T.Compose([
    T.Resize((DEDUP_SIZE, DEDUP_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])
JSON_PARSER = simdjson.Parser()

def oai_clip_image(x):
    return CLIP_IMAGE_TRANFORM(x)

def oai_clip_text(t):
    return CLIP_TEXT_TOKENIZER(t)[0]

def faces_scrfd(x):
    return FACES_IMAGE_TRANFORM(x)

def text_detection(x):
    return TEXT_DETECTION_IMAGE_TRANFORM(x)

def dedup(x):
    return DEDUP_IMAGE_TRANFORM(x)

def json_decoder(key, value, json_keys):

    if not key.endswith("json") or json_keys is None:
        return None

    json_dict = JSON_PARSER.parse(value).as_dict()

    return [json_dict[k] for k in json_keys]

def identity(a):
    return a