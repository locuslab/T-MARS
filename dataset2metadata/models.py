import abc
import logging
import torch.nn as nn
from dataset2metadata.text_detection.text_snake_wrapper import TextDetectorWrapper

logging.getLogger().setLevel(logging.INFO)


class WrapperMixin(metaclass=abc.ABCMeta):
    pass

class TextSnakeWrapper(nn.Module, WrapperMixin):

    name = 'texts-mmocr-snake'
    raw_inputs = ['image', ]
    preprocessors = ['texts-aug', ]
    dependencies = []
    to_device = True

    def __init__(self, device) -> None:
        super().__init__()
        self.model = TextDetectorWrapper(
        )
        logging.info(f'instantiated {self.name} on {device}')

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = x[0]
        return self.model(x)