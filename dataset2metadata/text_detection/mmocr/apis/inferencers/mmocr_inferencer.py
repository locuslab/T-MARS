# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import cv2
import mmcv
import mmengine
import numpy as np

from dataset2metadata.text_detection.mmocr.registry import VISUALIZERS
from dataset2metadata.text_detection.mmocr.structures.textdet_data_sample import TextDetDataSample
from dataset2metadata.text_detection.mmocr.utils import ConfigType, bbox2poly, crop_img, poly2bbox
from .base_mmocr_inferencer import (BaseMMOCRInferencer, InputsType, PredType,
                                    ResType)
from .kie_inferencer import KIEInferencer
from .textdet_inferencer import TextDetInferencer
from .textrec_inferencer import TextRecInferencer


class MMOCRInferencer(BaseMMOCRInferencer):

    def __init__(self,
                 det_config: Optional[Union[ConfigType, str]] = None,
                 det_ckpt: Optional[str] = None,
                 rec_config: Optional[Union[ConfigType, str]] = None,
                 rec_ckpt: Optional[str] = None,
                 kie_config: Optional[Union[ConfigType, str]] = None,
                 kie_ckpt: Optional[str] = None,
                 device: Optional[str] = None,
                 **kwargs) -> None:

        self.visualizer = None
        self.base_params = self._dispatch_kwargs(*kwargs)
        self.num_visualized_imgs = 0

        if det_config is not None:
            self.textdet_inferencer = TextDetInferencer(
                det_config, det_ckpt, device)
            self.mode = 'det'
        if rec_config is not None:
            self.textrec_inferencer = TextRecInferencer(
                rec_config, rec_ckpt, device)
            if getattr(self, 'mode', None) == 'det':
                self.mode = 'det_rec'
                ts = str(datetime.timestamp(datetime.now()))
                self.visualizer = VISUALIZERS.build(
                    dict(
                        type='TextSpottingLocalVisualizer',
                        name=f'inferencer{ts}',
                        font_families=self.textrec_inferencer.visualizer.
                        font_families))
            else:
                self.mode = 'rec'
        if kie_config is not None:
            if det_config is None or rec_config is None:
                raise ValueError(
                    'kie_config is only applicable when det_config and '
                    'rec_config are both provided')
            self.kie_inferencer = KIEInferencer(kie_config, kie_ckpt, device)
            self.mode = 'det_rec_kie'

    def preprocess(self, inputs: InputsType):
        new_inputs = []
        # import pdb;pdb.set_trace()
        for single_input in inputs[0]:
            # import pdb;pdb.set_trace()
            if isinstance(single_input, str):
                if osp.isdir(single_input):
                    raise ValueError('Feeding a directory is not supported')
                    # for img_path in os.listdir(single_input):
                    #     new_inputs.append(
                    #         mmcv.imread(osp.join(single_input, img_path)))
                else:
                    single_input = mmcv.imread(single_input)
                    # import pdb;pdb.set_trace()
                    new_inputs.append(single_input)
            else:
                new_inputs.append(single_input)
        # try:
        # print(new_inputs[0].shape)
        return new_inputs

    def forward(self, inputs: InputsType) -> PredType:
        """Forward the inputs to the model.

        Args:
            inputs (InputsType): The inputs to be forwarded.
        Returns:
            Dict: The prediction results. Possibly with keys "det", "rec", and
            "kie"..
        """
        result = {}
        result['det'] = self.textdet_inferencer(
            inputs, get_datasample=True)
        return result

    def visualize(self, inputs: InputsType, preds: PredType,
                  **kwargs) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[Dict]): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of images. Defaults to ''.
        """
        if 'kie' in self.mode:
            return self.kie_inferencer.visualize(self.kie_inputs, preds['kie'],
                                                 **kwargs)
        elif 'rec' in self.mode:
            if 'det' in self.mode:
                super().visualize(inputs, self._pack_e2e_datasamples(preds),
                                  **kwargs)
            else:
                return self.textrec_inferencer.visualize(
                    self.rec_inputs, preds['rec'][0], **kwargs)
        else:
            return self.textdet_inferencer.visualize(inputs, preds['det'],
                                                     **kwargs)

    def postprocess(self,
                    preds: PredType,
                    imgs: Optional[List[np.ndarray]] = None,
                    is_batch: bool = False,
                    inpt_img = None,
                    print_result: bool = False,
                    pred_out_file: str = ''
                    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Postprocess predictions.

        Args:
            preds (Dict): Predictions of the model.
            imgs (Optional[np.ndarray]): Visualized predictions.
            is_batch (bool): Whether the inputs are in a batch.
                Defaults to False.
            print_result (bool): Whether to print the result.
                Defaults to False.
            pred_out_file (str): Output file name to store predictions
                without images. Supported file formats are “json”, “yaml/yml”
                and “pickle/pkl”. Defaults to ''.

        Returns:
            Dict or List[Dict]: Each dict contains the inference result of
            each image. Possible keys are "det_polygons", "det_scores",
            "rec_texts", "rec_scores", "kie_labels", "kie_scores",
            "kie_edge_labels" and "kie_edge_scores".
        """

        results = [{} for _ in range(len(next(iter(preds.values()))))]
        if len(preds['det'])>1:
            raise ValueError('Batch implementation not supported yet')
        if 'det' in self.mode:
            control_points_filtered = []
            for i, det_pred in enumerate(preds['det']):
                det_dict_res = self.textdet_inferencer.pred2dict(det_pred)
                results[i].update(
                    dict(
                        det_polygons=det_dict_res['polygons'],
                        det_scores=det_dict_res['scores']))

                #filter out the ctrl points with a score less than 0.3
                # ctrl_point_per_image = np.asarray(det_dict_res['polygons'])
                scores = np.asarray(det_dict_res['scores'])
                # ctrl_point_filtered = ctrl_point_per_image[scores > 0.2]
                current_image = inpt_img[0][i]
                #overlay a polygon for each of the control points
                
                '''
                # we will need to set the image to bgr format
                compatible_image = np.copy(current_image)
                compatible_image = cv2.cvtColor(compatible_image, cv2.COLOR_RGB2BGR)
                '''
                compatible_image = None

                for j, ctrl_point in enumerate(det_dict_res['polygons']):
                    if scores[j]<0.3:
                        continue
                    #insert the control points into the list
                    #get the control points for the text prediction
                    ctrl_point = np.asarray(ctrl_point)
                    control_point_to_numpy = ctrl_point.reshape(-1, 2)
                    #save as list of control points
                    control_points_filtered.append(control_point_to_numpy.tolist())
                    control_point_to_numpy = control_point_to_numpy.astype(int)
                    continue
                    try:
                        #clamp the control points to the image size
                        # control_point_to_numpy = np.clip(control_point_to_numpy, 0, compatible_image.shape[0]-1)
                        #select color based on average of the polygon
                        control_point_to_numpy[:,0] = np.clip(control_point_to_numpy[:,0], 0, compatible_image.shape[1]-1)
                        control_point_to_numpy[:,1] = np.clip(control_point_to_numpy[:,1], 0, compatible_image.shape[0]-1)
                        color = current_image[control_point_to_numpy[:, 1], control_point_to_numpy[:, 0],:].mean(axis=0)
                        #invert to bgr format
                        color = color[::-1]
                        cv2.fillPoly(compatible_image, [control_point_to_numpy], color)
                    except:
                        print("Error in filling polygon id: ", id)

        is_batch = True
        return compatible_image, control_points_filtered

    def _pack_e2e_datasamples(self, preds: Dict) -> List[TextDetDataSample]:
        """Pack text detection and recognition results into a list of
        TextDetDataSample.

        Note that it is a temporary solution since the TextSpottingDataSample
        is not ready.
        """
        results = []
        for det_data_sample, rec_data_samples in zip(preds['det'],
                                                     preds['rec']):
            texts = []
            for rec_data_sample in rec_data_samples:
                texts.append(rec_data_sample.pred_text.item)
            det_data_sample.pred_instances.texts = texts
            results.append(det_data_sample)
        return results
