import torch
import torch.nn as nn
import numpy as np
from dataset2metadata.text_detection.mmocr.ocr import MMOCR
import cv2

counter = 0


class TextDetector(nn.Module):
    def __init__(self):
        super(TextDetector, self).__init__()        
        self.args = {"det": "TextSnake","det_config": None,"det_ckpt": None,"recog": None,"recog_config": None,"recog_ckpt": None,"kie": None,"kie_config": None,"kie_ckpt": None,"config_dir": "dataset2metadata/text_detection/mmocr/configs/","device": "cpu","show": False,"print_result": False,"pred_out_file": "","config_file_detectron2": "","confidence_threshold": 0.3,"opts": []}
        self.ocr = MMOCR(**self.args)
    
    def forward(self, x):
        global counter
        list_bounding_boxes_list = []
        for i in range(len(x)):
            img = x[i].transpose(1,2,0)
            img = np.expand_dims(img, axis=0)
            img = img*255
            img = img.astype('uint8')
            try:
                image, bounding_boxes_list = self.ocr.readtext(img = img, **self.args)
                # cv2 write image based on counter
                # cv2.imwrite("/home/pratyus2/tests2/"+str(counter)+".jpg", image)
            except:
                print("Skipping image of shape", img.shape)
                bounding_boxes_list = []

            list_bounding_boxes_list.append(bounding_boxes_list)
        
        #increase the counter and print it
        
        counter += 1

        return list_bounding_boxes_list


