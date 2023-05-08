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

counter = 0

class TextDetectorWrapper(nn.Module):
    def __init__(self, batch_size=512):
        super(TextDetectorWrapper, self).__init__()      
        cfg = Config.fromfile("config/fast/tt/fast_tiny_tt_512_finetune_ic17mlt.py") 
        cfg.batch_size = batch_size
        self.cfg = cfg
        checkpoint_path = "fast_tiny_tt_512_finetune_ic17mlt.pth"


        self.model = build_model(cfg.model)
        self.model = self.init_model(checkpoint_path)

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

    
    def forward(self, x):
        global counter

        batch_size = x.shape[0]
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
            contours = [(np.array(raw_contours[j]).reshape(-1,2)).tolist() for j in range(len(raw_contours))]
            all_contours.append(contours)
            all_texts.append("")

        counter += batch_size
        
        print(f"counter: {counter}")
        return all_contours, all_texts