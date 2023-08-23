import torch
import torch.nn as nn
import numpy as np
import cv2
from mmengine.config import Config
import sys
sys.path.append("dataset2metadata/text_detection")
sys.path.append("./")
from models import build_model
from models.utils import fuse_module, rep_model_convert

from mmocr.apis.inferencers.textrec_inferencer import TextRecInferencer
from mmocr.utils import bbox2poly, crop_img, poly2bbox


class TextDetectorWrapper(nn.Module):
    def __init__(self, batch_size=512):
        super(TextDetectorWrapper, self).__init__()      
        cfg = Config.fromfile("config/fast/tt/fast_tiny_tt_512_finetune_ic17mlt.py") 
        cfg.batch_size = batch_size
        self.cfg = cfg
        checkpoint_path = "fast_tiny_tt_512_finetune_ic17mlt.pth"

        self.model = build_model(cfg.model)
        self.model = self.init_model(checkpoint_path)
        self.textrec_inferencer = TextRecInferencer("CRNN", None, "cuda")

    def init_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['ema']
        d = dict()
        for key, value in state_dict.items():
            tmp = key.replace("module.", "")
            d[tmp] = value
        self.model.load_state_dict(d)
        self.model = self.model.to("cuda")

        self.model = rep_model_convert(self.model)
        # fuse conv and bn
        self.model = fuse_module(self.model)
        self.model.eval()
        return self.model

    
    def forward(self, x, get_text=False):
        global counter

        batch_size = x.shape[0]
        print(f"batch_size: {batch_size}")
        data = dict(imgs=x, 
                    img_metas={
                        'filename': [None for i in range(batch_size)],
                        'org_img_size': torch.ones((batch_size,2)).long()*512,
                        'img_size': torch.ones((batch_size,2)).long()*512,
            } 
        )
        data.update(dict(cfg=self.cfg))
        
        with torch.no_grad():
            outputs = self.model(**data)
        
        all_contours = []
        all_texts = []
        for i in range(batch_size):
            raw_contours = outputs["results"][i]["bboxes"]
            if get_text:
                img = x[i].cpu().numpy().transpose(1,2,0)
                
                crop_img_list = []
                for polygon in raw_contours:
                    quad = bbox2poly(poly2bbox(polygon)).tolist()
                    crop_img_list.append(crop_img(img, quad).astype('uint8'))
                
                if len(crop_img_list) > 0:
                    all_results = self.textrec_inferencer(crop_img_list,progress_bar=False)['predictions']
                    text = [all_results[i]["text"] for i in range(len(all_results))]
                    text = " ".join(text)
                    # print(text)
                    all_texts.append(text)
                else:
                    all_texts.append("")
            else:
                all_texts.append("")

            contours = [(np.array(raw_contours[j]).reshape(-1,2)).tolist() for j in range(len(raw_contours))]
            all_contours.append(contours)
            
        return all_contours, all_texts