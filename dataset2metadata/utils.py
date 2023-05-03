import hashlib
import os
import random
import urllib
import warnings
from typing import Dict, List, Set

import numpy as np
import torch
from tqdm import tqdm


def random_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def topsort(graph: Dict[str, List[str]]) -> List[str]:
    # from: https://stackoverflow.com/questions/52432988/python-dict-key-order-based-on-values-recursive-solution
    result: List[str] = []
    seen: Set[str] = set()

    def recursive_helper(node: str) -> None:
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                recursive_helper(neighbor)
        if node not in result:
            result.append(node)

    for key in graph.keys():
        recursive_helper(key)

    return result


def download(name: str, root: str = None):
    # modified from oai _download clip function

    if root is None:
        root = os.path.expanduser("~/.cache/dataset2metadata")

    cloud_checkpoints = {
        'nsfw-image': {
            # 'url': 'file://' + os.path.abspath('./assets/nsfw_torch.pt'),
            'url': 'https://github.com/mlfoundations/dataset2metadata/releases/download/v0.1.0-alpha/nsfw_torch.pt',
            'sha256': '3c97d5478477c181bfa29a33e6933f710c8ec587e3c3551ff855e293acdaf390',
        },
        'faces-scrfd10g': {
            # 'url': 'file://' + os.path.abspath('./assets/scrfd_10g.pt'),
            'url': 'https://github.com/mlfoundations/dataset2metadata/releases/download/v0.1.0-alpha/scrfd_10g.pt',
            'sha256': '963570df5e0ebf6bb313239d0f9f3f0c096c1ff6937e8e28e45abad4d8b1d5c7',
        },
        'dedup-embeddings': {
            # 'url': 'file://' + os.path.abspath('./assets/eval_dedup_embeddings.pt'),
            'url': 'https://github.com/mlfoundations/dataset2metadata/releases/download/v0.1.0-alpha/eval_dedup_embeddings.pt',
            'sha256': '18aeb08e1d53638761e05ac95254a2ef49653bfdb59a631c9304117ceef23d14',
        }
    }

    if name not in cloud_checkpoints:
        raise ValueError(f'unsupported cloud checkpoint: {name}. currently we only support: {list(cloud_checkpoints.keys())}')

    os.makedirs(root , exist_ok=True)

    expected_sha256 = cloud_checkpoints[name]['sha256']
    download_target = os.path.join(root, f'{name}.pt')
    url =  cloud_checkpoints[name]['url']

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def download_all():
    for k in ['nsfw-image', 'faces-scrfd10g', 'dedup-embeddings']:
        download(k)