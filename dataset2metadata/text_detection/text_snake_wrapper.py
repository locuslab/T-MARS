import torch
import torch.nn as nn
import numpy as np
# from dataset2metadata.text_detection.mmocr.mmocr.ocr import MMOCR
# from dataset2metadata.text_detection.mmocr.mmocr.apis.inferencers import MMOCRInferencer
import cv2
from dataset2metadata.text_detection.network.textnet import TextNet
from dataset2metadata.text_detection.util.detection import TextDetector
counter = 0


class TextDetectorWrapper(nn.Module):
    def __init__(self):
        super(TextDetectorWrapper, self).__init__()        
        self.model = TextNet(is_training=False, backbone='vgg')
        self.model.load_model("./dataset2metadata/text_detection/textsnake_vgg_180.pth")
        self.model = self.model.to("cuda")
        self.detector = TextDetector(self.model, tr_thresh=0.6, tcl_thresh=0.4)

    
    def forward(self, x):
        global counter
        # list_bounding_boxes_list = []
        list_text = []
        all_contours, _ = self.detector.detect(x)
        all_contours = [[contour.tolist() for contour in contours] for contours in all_contours]
        counter += x.shape[0]
        print(counter)
        return all_contours, list_text