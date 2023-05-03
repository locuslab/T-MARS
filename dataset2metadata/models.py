import abc
import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from clip import clip
from detoxify import Detoxify
from isc_feature_extractor import create_model
from dataset2metadata.utils import download
from dataset2metadata.face_detection.scrfd_wrapper import FaceDetector
from dataset2metadata.text_detection.text_snake_wrapper import TextDetector

logging.getLogger().setLevel(logging.INFO)



class WrapperMixin(metaclass=abc.ABCMeta):
    pass

class OaiClipWrapper(nn.Module, WrapperMixin, metaclass=abc.ABCMeta):

    def __init__(self, model_name: str, device: str) -> None:
        super().__init__()
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()

    def forward(self, x, t):
        image_feature = self.model.encode_image(x)
        text_feature = self.model.encode_text(t)

        # normalized features
        image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)

        return image_feature, text_feature

class OaiClipVitB32Wrapper(OaiClipWrapper):

    name = 'oai-clip-vit-b32'
    raw_inputs = ['image', 'text']
    preprocessors = ['clip-aug', 'clip-tokens']
    dependencies = []
    to_device = True

    def __init__(self, device) -> None:
        super().__init__('ViT-B/32', device)
        logging.info(f'instantiated {self.name} on {device}')

class OaiClipVitL14Wrapper(OaiClipWrapper):
    name = 'oai-clip-vit-l14'
    raw_inputs = ['image', 'text']
    preprocessors = ['clip-aug', 'clip-tokens']
    dependencies = []
    to_device = True

    def __init__(self, device) -> None:
        super().__init__('ViT-L/14', device)
        logging.info(f'instantiated {self.name} on {device}')

class DetoxifyWrapper(nn.Module, WrapperMixin):

    name = 'nsfw-detoxify'
    raw_inputs = ['text', ]
    preprocessors = ['identity', ]
    dependencies = []
    to_device = False

    def __init__(self, device) -> None:
        super().__init__()
        self.model = Detoxify('multilingual', device=f'cuda:{device}')
        logging.info(f'instantiated {self.name} on {device}')

    def forward(self, t):

        scores = []
        preds = self.model.predict(t)
        for _, v in preds.items():
            scores.append(v)

        # column-wise max score
        maxi, _  = torch.tensor(scores).transpose(0, 1).max(axis=1)

        return maxi

class NsfwImageWrapper(nn.Module, WrapperMixin):

    name = 'nsfw-image-oai-clip-vit-l-14'
    raw_inputs = []
    preprocessors = []
    dependencies = ['oai-clip-vit-l14']
    to_device = True

    def __init__(self, device) -> None:
        super().__init__()
        self.model = self.load(download('nsfw-image'))
        self.model.to(device).eval()
        logging.info(f'instantiated {self.name} on {device}')

    def load(self, classifier_path):
        names = ['norm', 'dense', 'relu', 'dense1', 'relu1', 'dense2', 'relu2', 'dense3', 'sigmoid']

        model = nn.Sequential(
            OrderedDict([
                (names[0], nn.BatchNorm1d(768, eps=0.0)),
                (names[1], nn.Linear(768, 64)),
                (names[2], nn.ReLU()),
                (names[3], nn.Linear(64, 512)),
                (names[4], nn.ReLU()),
                (names[5], nn.Linear(512, 256)),
                (names[6], nn.ReLU()),
                (names[7], nn.Linear(256, 1)),
                (names[8], nn.Sigmoid())
            ])
        )

        state_dict = torch.load(classifier_path, map_location='cpu')
        model.load_state_dict(state_dict)

        return model

    def forward(self, z):
        # use only the image feature
        return self.model(z[0].float()).squeeze()

class IscFtV107Wrapper(nn.Module, WrapperMixin):

    name = 'dedup-isc-ft-v107'
    raw_inputs = ['image', ]
    preprocessors = ['dedup-aug', ]
    dependencies = []
    to_device = True

    def __init__(self, device) -> None:
        super().__init__()

        self.model, _ = create_model(weight_name='isc_ft_v107', device=device)
        self.model.eval()
        self.reference_embeddings = torch.load(download('dedup-embeddings')).to(device).t()
        self.reference_embeddings.requires_grad = False
        logging.info(f'instantiated {self.name} on {device}')

    def forward(self, x):
        z = self.model(x)
        z /= z.norm(dim=-1, keepdim=True)
        scores = z @ self.reference_embeddings # (b,c) @ (c,n) = (b,n)
        max_scores, _ = scores.max(axis=1)

        return z, max_scores

class Scrfd10GWrapper(nn.Module, WrapperMixin):

    name = 'faces-scrfd10g'
    raw_inputs = ['image', ]
    preprocessors = ['faces-aug', ]
    dependencies = []
    to_device = True

    def __init__(self, device) -> None:
        super().__init__()
        self.model = FaceDetector(
            download('faces-scrfd10g'),
            device
        )
        logging.info(f'instantiated {self.name} on {device}')

    def forward(self, x):
        return self.model.detect_faces(images=x[0], paddings=x[1])


class TextSnakeWrapper(nn.Module, WrapperMixin):

    name = 'texts-mmocr-snake'
    raw_inputs = ['image', ]
    preprocessors = ['texts-aug', ]
    dependencies = []
    to_device = True

    def __init__(self, device) -> None:
        super().__init__()
        self.model = TextDetector(
        )
        logging.info(f'instantiated {self.name} on {device}')

    def forward(self, x):
        x = x.cpu().numpy()
        return self.model(x)