# Originally edited from https://github.com/PUTvision/qgis-plugin-deepness

import numpy as np
from typing import List
from .utils import xywh2xyxy
from .ckdtree import cKDTree

def preprocess(model_input):
    s = model_input.shape
    if not len(s) in [3,4]:
        raise Exception(f"Expected input with 3 or 4 dimensions, got: {s}")
    is_batched = len(s) == 4

    # expected: [batch],channel,height,width but could be: [batch],height,width,channel
    if s[-1] in [3,4] and s[1] > s[-1]:
        if is_batched:
            model_input = np.transpose(model_input, (0, 3, 1, 2))
        else:
            model_input = np.transpose(model_input, (2, 0, 1))
    
    # add batch dimension (1, c, h, w)
    if not is_batched:
        model_input = np.expand_dims(model_input, axis=0)
    
    # drop alpha channel
    if model_input.shape[1] == 4:
        model_input = model_input[:,0:3,:,:]
    
    if model_input.shape[1] != 3:
        raise Exception(f"Expected input channels to be 3, but got: {model_input.shape[1]}")
    
    # normalize
    if model_input.dtype == np.uint8:
        return (model_input / 255.0).astype(np.float32)
    
    if model_input.dtype.kind == 'f':
        min_value = float(model_input.min())
        value_range = float(model_input.max()) - min_value
    else:
        data_range = np.iinfo(image.dtype)
        min_value = 0
        value_range = float(data_range.max) - float(data_range.min)
    
    model_input = model_input.astype(np.float32)
    model_input -= min_value
    model_input /= value_range
    model_input[model_input > 1] = 1
    model_input[model_input < 0] = 0

    return model_input

def postprocess(model_output, config):
    filtered = model_output[model_output[:, :, 4] >= config['det_conf']]
    if not len(filtered):
        return np.empty((0, 6), dtype=np.float32)
    
    outputs = xywh2xyxy(filtered)

    return non_max_suppression_fast(outputs, config['det_iou_thresh'])

def extract_bsc(outputs):
    boxes = outputs[:, :4].astype(np.int32)
    scores = outputs[:, 4]
    classes = np.argmax(outputs[:, 5:], axis=1)

    return boxes, scores, classes

def compute_centers(outputs):
    return np.array([(outputs[:,0] + outputs[:,2]) / 2, (outputs[:,1] + outputs[:,3]) / 2]).T

def compute_areas(outputs):
    return np.array([(outputs[:,2] - outputs[:,0]) * (outputs[:,3] - outputs[:,1])]).flatten()

def sort_by_area(outputs, reverse=False):
    areas = compute_areas(outputs)
    if reverse:
        areas *= -1
    return outputs[np.argsort(areas)]

def non_max_suppression_fast(outputs: np.ndarray, iou_threshold: float) -> List:
    """Apply non-maximum suppression to bounding boxes

    Based on:
    https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py

    Parameters
    ----------
    outputs : np.ndarray
        Output array containing bounding boxes in (x1,y1,x2,y2) format and scores
    iou_threshold : float
        IoU threshold

    Returns
    -------
    np.ndarray
        Bounding boxes to keep
    """
    # If no bounding boxes, return empty list
    if len(outputs) == 0:
        return []

    scores = outputs[:, 4]

    # coordinates of bounding boxes
    start_x = outputs[:, 0]
    start_y = outputs[:, 1]
    end_x = outputs[:, 2]
    end_y = outputs[:, 3]

    # Picked bounding boxes
    pick_ids = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(scores)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        pick_ids.append(index)
        ratio = compute_iou(index, order, start_x, start_y, end_x, end_y, areas)

        left = np.where(ratio < iou_threshold)
        order = order[left]

    return outputs[pick_ids]


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


def non_max_kdtree(outputs: np.ndarray, iou_threshold: float) -> np.ndarray:
    """ Remove overlapping bounding boxes using kdtree

    :param outputs: array of bounding boxes in (xyxy format)
    :param iou_threshold: Threshold for intersection over union
    :return: Filtered output
    """

    centers = compute_centers(outputs)
    bboxes = outputs[:, :4]
    areas = compute_areas(outputs)

    kdtree = cKDTree(centers)
    pick_ids = set()
    removed_ids = set()

    for i, out in enumerate(outputs):
        if i in removed_ids:
            continue
        
        indices = kdtree.query(centers[i], k=min(10, len(outputs)))

        for j in indices:
            if j in removed_ids:
                continue

            if i == j:
                continue
            
            # x_min,y_min,x_max,y_max
            x_min = max(bboxes[i][0], bboxes[j][0])
            y_min = max(bboxes[i][1], bboxes[j][1])
            x_max = min(bboxes[i][2], bboxes[j][2])
            y_max = min(bboxes[i][3], bboxes[j][3])
            
            iou = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1) / min(areas[i], areas[j])

            if iou > iou_threshold:
                removed_ids.add(j)

        pick_ids.add(i)
    print(removed_ids, pick_ids)
    return outputs[np.asarray(list(pick_ids))]


def execute(images, session, config):
    is_batched = len(images.shape) == 4
    images = preprocess(images)
    outs = session.run(None, {config['input_name']: images})

    results = []
    for out in outs:
        results.append(postprocess(out, config))
    
    if is_batched:
        return np.asarray(results)
    else:
        return results[0]
