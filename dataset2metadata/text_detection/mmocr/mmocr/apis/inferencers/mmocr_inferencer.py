# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import mmengine
import numpy as np
from rich.progress import track

from mmocr.registry import VISUALIZERS
from mmocr.structures import TextSpottingDataSample
from mmocr.utils import ConfigType, bbox2poly, crop_img, poly2bbox
from .base_mmocr_inferencer import (BaseMMOCRInferencer, InputsType, PredType,
                                    ResType)
from .kie_inferencer import KIEInferencer
from .textdet_inferencer import TextDetInferencer
from .textrec_inferencer import TextRecInferencer


class MMOCRInferencer(BaseMMOCRInferencer):
    """MMOCR Inferencer. It's a wrapper around three base task
    inferenecers: TextDetInferencer, TextRecInferencer and KIEInferencer,
    and it can be used to perform end-to-end OCR or KIE inference.

    Args:
        det (Optional[Union[ConfigType, str]]): Pretrained text detection
            algorithm. It's the path to the config file or the model name
            defined in metafile. Defaults to None.
        det_weights (Optional[str]): Path to the custom checkpoint file of
            the selected det model. If it is not specified and "det" is a model
            name of metafile, the weights will be loaded from metafile.
            Defaults to None.
        rec (Optional[Union[ConfigType, str]]): Pretrained text recognition
            algorithm. It's the path to the config file or the model name
            defined in metafile. Defaults to None.
        rec_weights (Optional[str]): Path to the custom checkpoint file of
            the selected rec model. If it is not specified and "rec" is a model
            name of metafile, the weights will be loaded from metafile.
            Defaults to None.
        kie (Optional[Union[ConfigType, str]]): Pretrained key information
            extraction algorithm. It's the path to the config file or the model
            name defined in metafile. Defaults to None.
        kie_weights (Optional[str]): Path to the custom checkpoint file of
            the selected kie model. If it is not specified and "kie" is a model
            name of metafile, the weights will be loaded from metafile.
            Defaults to None.
        device (Optional[str]): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.

    """

    def __init__(self,
                 det: Optional[Union[ConfigType, str]] = None,
                 det_weights: Optional[str] = None,
                 rec: Optional[Union[ConfigType, str]] = None,
                 rec_weights: Optional[str] = None,
                 kie: Optional[Union[ConfigType, str]] = None,
                 kie_weights: Optional[str] = None,
                 device: Optional[str] = None) -> None:

        if det is None and rec is None and kie is None:
            raise ValueError('At least one of det, rec and kie should be '
                             'provided.')

        self.visualizer = None

        if det is not None:
            self.textdet_inferencer = TextDetInferencer(
                det, det_weights, device)
            self.mode = 'det'
        if rec is not None:
            self.textrec_inferencer = TextRecInferencer(
                rec, rec_weights, device)
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
        if kie is not None:
            if det is None or rec is None:
                raise ValueError(
                    'kie_config is only applicable when det_config and '
                    'rec_config are both provided')
            self.kie_inferencer = KIEInferencer(kie, kie_weights, device)
            self.mode = 'det_rec_kie'

    def _inputs2ndarrray(self, inputs: List[InputsType]) -> List[np.ndarray]:
        """Preprocess the inputs to a list of numpy arrays."""
        new_inputs = []
        for item in inputs:
            if isinstance(item, np.ndarray):
                new_inputs.append(item)
            elif isinstance(item, str):
                img_bytes = mmengine.fileio.get(item)
                new_inputs.append(mmcv.imfrombytes(img_bytes))
            else:
                raise NotImplementedError(f'The input type {type(item)} is not'
                                          'supported yet.')
        return new_inputs

    def forward(self,
                inputs: InputsType,
                batch_size: int = 1,
                det_batch_size: Optional[int] = None,
                rec_batch_size: Optional[int] = None,
                kie_batch_size: Optional[int] = None,
                **forward_kwargs) -> PredType:
        """Forward the inputs to the model.

        Args:
            inputs (InputsType): The inputs to be forwarded.
            batch_size (int): Batch size. Defaults to 1.
            det_batch_size (Optional[int]): Batch size for text detection
                model. Overwrite batch_size if it is not None.
                Defaults to None.
            rec_batch_size (Optional[int]): Batch size for text recognition
                model. Overwrite batch_size if it is not None.
                Defaults to None.
            kie_batch_size (Optional[int]): Batch size for KIE model.
                Overwrite batch_size if it is not None.
                Defaults to None.

        Returns:
            Dict: The prediction results. Possibly with keys "det", "rec", and
            "kie"..
        """
        result = {}
        forward_kwargs['progress_bar'] = False
        if det_batch_size is None:
            det_batch_size = batch_size
        if rec_batch_size is None:
            rec_batch_size = batch_size
        if kie_batch_size is None:
            kie_batch_size = batch_size
        if self.mode == 'rec':
            # The extra list wrapper here is for the ease of postprocessing
            self.rec_inputs = inputs
            predictions = self.textrec_inferencer(
                self.rec_inputs,
                return_datasamples=True,
                batch_size=rec_batch_size,
                **forward_kwargs)['predictions']
            result['rec'] = [[p] for p in predictions]
        elif self.mode.startswith('det'):  # 'det'/'det_rec'/'det_rec_kie'
            result['det'] = self.textdet_inferencer(
                inputs,
                return_datasamples=True,
                batch_size=det_batch_size,
                **forward_kwargs)['predictions']
            if self.mode.startswith('det_rec'):  # 'det_rec'/'det_rec_kie'
                result['rec'] = []
                for img, det_data_sample in zip(
                        self._inputs2ndarrray(inputs), result['det']):
                    det_pred = det_data_sample.pred_instances
                    self.rec_inputs = []
                    for polygon in det_pred['polygons']:
                        # Roughly convert the polygon to a quadangle with
                        # 4 points
                        quad = bbox2poly(poly2bbox(polygon)).tolist()
                        self.rec_inputs.append(crop_img(img, quad))
                    result['rec'].append(
                        self.textrec_inferencer(
                            self.rec_inputs,
                            return_datasamples=True,
                            batch_size=rec_batch_size,
                            **forward_kwargs)['predictions'])
                if self.mode == 'det_rec_kie':
                    self.kie_inputs = []
                    # TODO: when the det output is empty, kie will fail
                    # as no gt-instances can be provided. It's a known
                    # issue but cannot be solved elegantly since we support
                    # batch inference.
                    for img, det_data_sample, rec_data_samples in zip(
                            inputs, result['det'], result['rec']):
                        det_pred = det_data_sample.pred_instances
                        kie_input = dict(img=img)
                        kie_input['instances'] = []
                        for polygon, rec_data_sample in zip(
                                det_pred['polygons'], rec_data_samples):
                            kie_input['instances'].append(
                                dict(
                                    bbox=poly2bbox(polygon),
                                    text=rec_data_sample.pred_text.item))
                        self.kie_inputs.append(kie_input)
                    result['kie'] = self.kie_inferencer(
                        self.kie_inputs,
                        return_datasamples=True,
                        batch_size=kie_batch_size,
                        **forward_kwargs)['predictions']
        return result

    def visualize(self, inputs: InputsType, preds: PredType,
                  **kwargs) -> Union[List[np.ndarray], None]:
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
            save_vis (bool): Whether to save the visualization result. Defaults
                to False.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """

        if 'kie' in self.mode:
            return self.kie_inferencer.visualize(self.kie_inputs, preds['kie'],
                                                 **kwargs)
        elif 'rec' in self.mode:
            if 'det' in self.mode:
                return super().visualize(inputs,
                                         self._pack_e2e_datasamples(preds),
                                         **kwargs)
            else:
                return self.textrec_inferencer.visualize(
                    self.rec_inputs, preds['rec'][0], **kwargs)
        else:
            return self.textdet_inferencer.visualize(inputs, preds['det'],
                                                     **kwargs)

    def __call__(
        self,
        inputs: InputsType,
        batch_size: int = 1,
        det_batch_size: Optional[int] = None,
        rec_batch_size: Optional[int] = None,
        kie_batch_size: Optional[int] = None,
        out_dir: str = 'results/',
        return_vis: bool = False,
        save_vis: bool = False,
        save_pred: bool = False,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer. It can be a path
                to image / image directory, or an array, or a list of these.
            batch_size (int): Batch size. Defaults to 1.
            det_batch_size (Optional[int]): Batch size for text detection
                model. Overwrite batch_size if it is not None.
                Defaults to None.
            rec_batch_size (Optional[int]): Batch size for text recognition
                model. Overwrite batch_size if it is not None.
                Defaults to None.
            kie_batch_size (Optional[int]): Batch size for KIE model.
                Overwrite batch_size if it is not None.
                Defaults to None.
            out_dir (str): Output directory of results. Defaults to 'results/'.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            save_vis (bool): Whether to save the visualization results to
                "out_dir". Defaults to False.
            save_pred (bool): Whether to save the inference results to
                "out_dir". Defaults to False.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results, mapped from
                "predictions" and "visualization".
        """
        if (save_vis or save_pred) and not out_dir:
            raise ValueError('out_dir must be specified when save_vis or '
                             'save_pred is True!')
        if out_dir:
            img_out_dir = osp.join(out_dir, 'vis')
            pred_out_dir = osp.join(out_dir, 'preds')
        else:
            img_out_dir, pred_out_dir = '', ''

        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(
            save_vis=save_vis,
            save_pred=save_pred,
            return_vis=return_vis,
            **kwargs)

        ori_inputs = self._inputs_to_list(inputs)
        if det_batch_size is None:
            det_batch_size = batch_size
        if rec_batch_size is None:
            rec_batch_size = batch_size
        if kie_batch_size is None:
            kie_batch_size = batch_size

        chunked_inputs = super(BaseMMOCRInferencer,
                               self)._get_chunk_data(ori_inputs, batch_size)
        results = {'predictions': [], 'visualization': []}
        # for ori_input in track(chunked_inputs, description='Inference'):
        for ori_input in chunked_inputs:
            preds = self.forward(
                ori_input,
                det_batch_size=det_batch_size,
                rec_batch_size=rec_batch_size,
                kie_batch_size=kie_batch_size,
                **forward_kwargs)
            visualization = self.visualize(
                ori_input, preds, img_out_dir=img_out_dir, **visualize_kwargs)
            # import pdb; pdb.set_trace()
            batch_res = self.postprocess(
                ori_input,
                preds,
                visualization,
                pred_out_dir=pred_out_dir,
                **postprocess_kwargs)
            # results['predictions'].extend(batch_res['predictions'])
            # if return_vis and batch_res['visualization'] is not None:
            #     results['visualization'].extend(batch_res['visualization'])
        return batch_res

    def postprocess(self,
                    ori_input,
                    preds: PredType,
                    visualization: Optional[List[np.ndarray]] = None,
                    print_result: bool = False,
                    save_pred: bool = False,
                    pred_out_dir: str = ''
                    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (PredType): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.
            print_result (bool): Whether to print the result.
                Defaults to False.
            save_pred (bool): Whether to save the inference result. Defaults to
                False.
            pred_out_dir: File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            Dict: Inference and visualization results, mapped from
                "predictions" and "visualization".
        """

        results = [{} for _ in range(len(next(iter(preds.values()))))]
        if len(preds['det'])>1:
            raise ValueError('Batch implementation not supported yet')
        
        batch_list_text = []
        if 'rec' in self.mode:
            for i, rec_pred in enumerate(preds['rec']):
                list_text = []
                # result = dict(rec_texts=[], rec_scores=[])
                for rec_pred_instance in rec_pred:
                    rec_dict_res = self.textrec_inferencer.pred2dict(
                        rec_pred_instance)
                    # if rec_dict_res['scores']>0.3:
                    list_text.append(rec_dict_res['text'])
                list_text = [" ".join(list_text)]
                batch_list_text.append(list_text)
                # pred_results[i].update(result)

        batch_control_points_filtered = []
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
                current_image = ori_input[0][i]
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
                batch_control_points_filtered.append(control_points_filtered)
        is_batch = True
        return compatible_image, batch_control_points_filtered, batch_list_text

    def _pack_e2e_datasamples(self,
                              preds: Dict) -> List[TextSpottingDataSample]:
        """Pack text detection and recognition results into a list of
        TextSpottingDataSample."""
        results = []

        for det_data_sample, rec_data_samples in zip(preds['det'],
                                                     preds['rec']):
            texts = []
            for rec_data_sample in rec_data_samples:
                texts.append(rec_data_sample.pred_text.item)
            det_data_sample.pred_instances.texts = texts
            results.append(det_data_sample)
        return results
