# Originally edited from https://github.com/PUTvision/qgis-plugin-deepness

import numpy as np
from typing import List
from .utils import xywh2xyxy

def postprocess(model_output, config):
    filtered = model_output[model_output[:, :, 4] >= config['det_conf']]
    if not len(filtered):
        return [], [], []
    
    scores = filtered[:, 4]
    outputs = xywh2xyxy(filtered)
    outputs_nms = outputs[non_max_suppression_fast(outputs, scores, config['det_iou_thresh'])]
    
    boxes = outputs_nms[:, :4].astype(np.int32)
    scores = outputs_nms[:, 4]
    classes = np.argmax(outputs_nms[:, 5:], axis=1)

    return boxes, scores, classes

def non_max_suppression_fast(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List:
    """Apply non-maximum suppression to bounding boxes

    Based on:
    https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py

    Parameters
    ----------
    boxes : np.ndarray
        Bounding boxes in (x1,y1,x2,y2) format
    scores : np.ndarray
        Confidence scores
    iou_threshold : float
        IoU threshold

    Returns
    -------
    List
        List of indexes of bounding boxes to keep
    """
    # If no bounding boxes, return empty list
    if len(boxes) == 0:
        return []

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Picked bounding boxes
    picked_boxes = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(scores)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(index)
        ratio = compute_iou(index, order, start_x, start_y, end_x, end_y, areas)

        left = np.where(ratio < iou_threshold)
        order = order[left]

    return picked_boxes


def compute_iou(index: int, order: np.ndarray, start_x: np.ndarray, start_y: np.ndarray, end_x: np.ndarray, end_y: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Compute IoU for bounding boxes
    
    Parameters
    ----------
    index : int
        Index of the bounding box
    order : np.ndarray
        Order of bounding boxes
    start_x : np.ndarray
        Start x coordinate of bounding boxes
    start_y : np.ndarray
        Start y coordinate of bounding boxes
    end_x : np.ndarray
        End x coordinate of bounding boxes
    end_y : np.ndarray
        End y coordinate of bounding boxes
    areas : np.ndarray
        Areas of bounding boxes

    Returns
    -------
    np.ndarray
        IoU values
    """
    
    # Compute ordinates of intersection-over-union(IOU)
    x1 = np.maximum(start_x[index], start_x[order[:-1]])
    x2 = np.minimum(end_x[index], end_x[order[:-1]])
    y1 = np.maximum(start_y[index], start_y[order[:-1]])
    y2 = np.minimum(end_y[index], end_y[order[:-1]])

    # Compute areas of intersection-over-union
    w = np.maximum(0.0, x2 - x1 + 1)
    h = np.maximum(0.0, y2 - y1 + 1)
    intersection = w * h

    # Compute the ratio between intersection and union
    return intersection / (areas[index] + areas[order[:-1]] - intersection)
