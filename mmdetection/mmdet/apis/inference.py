import warnings

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.models import build_detector
from imantics import Polygons, Mask
import cv2

def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['classes']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, device)
    else:
        return _inference_generator(model, imgs, img_transform, device)


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, model.cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, device)


# TODO: merge this method with the one in BaseDetector
def show_result(img, result, class_names, score_thr=0.3, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            colors = {k: [] for k in 'rgb'}
            temp = {k: np.random.randint(0, 255) for k in 'rgb'}
            for k in temp:
                while 1:
                    c = temp[k]
                    t = set(j for j in range(c - 25, c + 25) if 0 <= j <= 255)
                    if t.intersection(colors[k]):
                        temp[k] = np.random.randint(0, 255)
                    else:
                        break
                colors[k].append(temp[k])
            color_mask = np.array([colors['r'][0], colors['g'][0], colors['b'][0]])
            # color_mask = np.random.randint(
            #     0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    thickness_t = 2 if img.shape[0] < 3000 else 6
    font_scale_t = 0.65 if img.shape[0] < 3000 else 2.5
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        text_color='yellow',
        thickness=thickness_t,
        font_scale=font_scale_t,
        show=out_file is None,
        out_file=out_file,
        win_name='demo')

def result2dict(img, result, class_names, score_thr=0.3, out_file=None):

    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    arr_poly = []
    arr_masks = []
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            # img[mask] = img[mask] * 0.5 + color_mask * 0.5
            polygons = Mask(mask).polygons()
            if polygons.points:
                arr_poly.append(polygons.points[0])
                arr_masks.append(mask)
            else:
                arr_poly.append(np.empty([]))
                arr_masks.append(np.empty([]))

    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    if not arr_poly:
        return None
    keep = np.array([i for i, val in enumerate(arr_poly) if (val != np.empty([])).any()])
    labels = np.concatenate(labels)
    ret_poly = [i.tolist() for i in np.array(arr_poly)[keep]]
    ret_mask = [i.tolist() for i in np.array(arr_masks)[keep]]
    return bboxes[keep].tolist(), ret_poly, np.array([class_names[i] for i in labels])[keep].tolist(), ret_mask

    # mmcv.imshow_det_bboxes(
    #     img.copy(),
    #     bboxes,
    #     labels,
    #     class_names=class_names,
    #     score_thr=score_thr,
    #     show=out_file is None,
    #     out_file=out_file)


