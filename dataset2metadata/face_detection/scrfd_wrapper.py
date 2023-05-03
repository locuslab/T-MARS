# contributed by George Smyrnis

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from dataset2metadata.face_detection.backbone import ResNetV1e
from dataset2metadata.face_detection.head import SCRFDHead
from dataset2metadata.face_detection.neck import PAFPN
from dataset2metadata.face_detection.utils import bbox2result


class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.
        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.
        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image metas ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.
        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.
        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.
        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.
                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.
        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.
    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head={},
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = ResNetV1e(**backbone)
        if neck is not None:
            self.neck = PAFPN(**neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = SCRFDHead(**bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.
        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if torch.onnx.is_in_onnx_export():
            print('single_stage.py in-onnx-export')
            print(outs.__class__)
            cls_score, bbox_pred = outs
            for c in cls_score:
                print(c.shape)
            for c in bbox_pred:
                print(c.shape)
            return outs
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.
        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'
        print('aug-test:', len(imgs))
        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]


class SCRFD(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SCRFD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypointss=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_keypointss, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.
        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if torch.onnx.is_in_onnx_export():
            print('single_stage.py in-onnx-export')
            print(outs.__class__)
            cls_score, bbox_pred, kps_pred = outs
            for c in cls_score:
                print(c.shape)
            for c in bbox_pred:
                print(c.shape)
            if self.bbox_head.use_kps:
                for c in kps_pred:
                    print(c.shape)
                return (cls_score, bbox_pred, kps_pred)
            else:
                return (cls_score, bbox_pred)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def feature_test(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs


MODEL_CONFIG = dict(
    type='SCRFD',
    backbone=dict(
        depth=0,
        block_cfg=dict(
            block='BasicBlock',
            stage_blocks=(3, 4, 2, 3),
            stage_planes=[56, 88, 88, 224]),
        base_channels=56,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        in_channels=[56, 88, 88, 224],
        out_channels=56,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3),
    bbox_head=dict(
        num_classes=1,
        in_channels=56,
        stacked_convs=3,
        feat_channels=80,
        #norm_cfg=dict(type='BN', requires_grad=True),
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
        cls_reg_share=True,
        strides_share=True,
        scale_mode=2,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales = [1,2],
            base_sizes = [16, 64, 256],
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=False,
        reg_max=8,
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0),
        use_kps=False,
        loss_kps=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.1),
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=-1,
            min_bbox_size=0,
            score_thr=0.02,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=-1)))

TEST_CFG = dict(
    nms_pre=-1,
    min_bbox_size=0,
    score_thr=0.02,
    nms=dict(type='nms', iou_threshold=0.45),
    max_per_img=-1)

def scrfd_wrapper(checkpoint_path):
    model = SCRFD(MODEL_CONFIG["backbone"], MODEL_CONFIG["neck"], MODEL_CONFIG["bbox_head"], test_cfg=TEST_CFG)
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)

    return model

class FaceDetector(object):

    def __init__(
        self,
        checkpoint_path: str,
        device: int = 0
    ) -> None:

        if device < 0:
            self.device = 'cpu'
        else:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    'CUDA is not available. Consider setting gpu = -1.'
                )
            else:
                self.device = f'cuda:{device}'

        self.model = scrfd_wrapper(checkpoint_path).eval().to(self.device)

    def _apply_threshold(
        self,
        bbox_result_image,
        thr: float
    ) -> torch.Tensor:
        """Apply threshold to predicted bboxes.

        Args:
            bbox_result_image: Results on bboxes as outputted by the model
                bbox_head. A tuple of tensors, with the first being the tensor
                containing possible face detections and scores.
            thr: Threshold to apply on scores.

        Returns:
            A (F,4) tensor, of type torch.float32, containing the indices of the
                bounding box corners, in the format [x_top_left, y_top_left,
                x_bottom_right, y_bottom_right]. The indices are scaled from
                0 to 1 (corresponding to a relative point from the top left
                part of the image).
        """
        bboxes_with_score = bbox_result_image[0]
        scores = bboxes_with_score[:, -1]
        ind = scores > thr
        predicted_bboxes = bboxes_with_score[ind, :-1]
        return predicted_bboxes

    def _rescale(
        self,
        bbox: torch.Tensor,
        paddings: torch.Tensor,
        img_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Rescale bounding box to [0,1].

        Args:
            bbox: Bounding box to rescale.
            img_shape: Dimensions (H,W) of image.
            paddings: The pads in each border of the image.

        Returns:
            A torch.Tensor corresponding to the bounding box of the image, with
                elements in [0,1] (relative positions with respect to the top
                left corner of the original image).
        """
        H, W = img_shape
        result = torch.zeros_like(bbox).type(torch.float32)

        pad_left, pad_top, pad_right, pad_bottom = paddings

        result[:, 0] = torch.clamp((bbox[:,0] - pad_left * W) / ((1.0 - pad_left - pad_right) * W), 0, 1)
        result[:, 1] = torch.clamp((bbox[:,1] - pad_top * H) / ((1.0 - pad_top - pad_bottom) * H), 0, 1)
        result[:, 2] = torch.clamp((bbox[:,2] - pad_left * W) / ((1.0 - pad_left - pad_right) * W), 0, 1)
        result[:, 3] = torch.clamp((bbox[:,3] - pad_top * H) / ((1.0 - pad_top - pad_bottom) * H), 0, 1)

        return result

    def detect_faces(
        self,
        images: torch.Tensor,
        paddings: torch.Tensor,
        score_threshold: float = 0.3,
    ):
        """Detect faces.

        Args:
            images: Batch of images, of shape (N, C, H, W).
            paddings: Pads in each image, of shape (N, 4).
            score_threshold: Confidence cutoff for bounding boxes, in [0,1].
                Default = 0.3

        Returns:
            A list of list of float, with each element corresponding to a single
                image. For each element there are F inner lists with 4 elements,
                where F is the number of faces detected in the image with format
                [x_top_left, y_top_left, x_bottom_right, y_bottom_right].
        """
        with torch.no_grad():
            if len(images.shape) != 4 or images.shape[1] != 3:
                raise ValueError(
                    'Images should be provided as a tensor of shape (N,3,H,W).'
                )

            if images.dtype == torch.uint8:
                images = images.float() / 255.0

            images = images.to(self.device)

            N, C, H, W = images.shape

            img_metas = [{'img_shape': (H, W, C), 'scale_factor': 1}] * N

            result = self.model.feature_test(images)    # type: ignore
            bboxes = self.model.bbox_head.get_bboxes(   # type: ignore
                *result, img_metas=img_metas
            )

            # Threshold the bounding boxes.
            bboxes = list(map(
                lambda x: self._apply_threshold(x, score_threshold), bboxes
            ))

            # Make the bounding boxes relative to image.
            bboxes = list(map(
                lambda x, paddings: self._rescale(x, paddings, (H,W)), bboxes, paddings.unbind()
            ))

            # Make the result be a list of lists.
            bboxes = list(map(
                lambda x: x.tolist(), bboxes
            ))

            return bboxes