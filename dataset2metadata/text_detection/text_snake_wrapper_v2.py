import torch
import torch.nn as nn
import numpy as np
# from dataset2metadata.text_detection.mmocr.mmocr.ocr import MMOCR
from dataset2metadata.text_detection.mmocr.mmocr.apis.inferencers import MMOCRInferencer
import cv2

counter = 0


class TextDetector(nn.Module):
    def __init__(self):
        super(TextDetector, self).__init__()        
        self.args = {"det": "TextSnake","rec": "ASTER","device": "cpu"}
        self.ocr = MMOCRInferencer(**self.args)
    
    def forward(self, x):
        global counter
        list_bounding_boxes_list = []
        for i in range(1):
            img = x.transpose(0,2,3,1)
            
            # img = np.expand_dims(img, axis=0)
            img = img*255
            img = img.astype('uint8')
            # try:
            # import pdb; pdb.set_trace()
            image, bounding_boxes_list, list_text = self.ocr(inputs = img, batch_size = 2)
                # list_test = " ".join(list_text)
                #the list is of tyeps [a,b,c]
                #convert it to [a b c]
                
            #print(counter)
            print(list_text)
            # cv2 write image based on counter
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("/data/testing/mask/"+str(counter)+".jpg", img)
            # except:
            #      print("Skipping image of shape", img.shape)
            #      bounding_boxes_list = []

            list_bounding_boxes_list.append(bounding_boxes_list)
        
        #increase the counter and print it
        
        counter += img.shape[0]
        print(counter)

        return list_bounding_boxes_list, list_text


