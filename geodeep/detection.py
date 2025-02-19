# Originally edited from https://github.com/PUTvision/qgis-plugin-deepness
import rasterio
import rasterio.warp
import numpy as np
import json
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
        data_range = np.iinfo(model_input.dtype)
        min_value = 0
        value_range = float(data_range.max) - float(data_range.min)
    
    model_input = model_input.astype(np.float32)
    model_input -= min_value
    model_input /= value_range
    model_input[model_input > 1] = 1
    model_input[model_input < 0] = 0

    return model_input

def postprocess(model_output, config):
    if config['det_type'] in ["YOLO_v9", "YOLO_v8"]:
        model_output = np.transpose(model_output, (0, 2, 1))

    if config['det_type'] in ["YOLO_v9", "YOLO_v8"]:
        filtered = model_output[np.max(model_output[:, :, 4:], axis=2) >= config['det_conf']]
    else:
        filtered = model_output[model_output[:, :, 4] >= config['det_conf']]
    
    if not len(filtered):
        return np.empty((0, model_output.shape[-1]), dtype=np.float32)
    
    if config['det_type'] == 'retinanet':
        bscs = filtered
    else:
        bscs = xywh2xyxy(filtered)
    
    if len(config['classes']) > 0:
        classes = extract_classes(bscs, config)
        bscs = bscs[np.isin(classes, config['classes'])]

    return non_max_suppression_fast(bscs, config)

def extract_bsc(bscs, config):
    if not len(bscs):
        return [], [], []
    
    boxes = bscs[:, :4].astype(np.int32)
    scores = extract_scores(bscs, config)
    classes = extract_classes(bscs, config)
    
    return boxes.tolist(), scores.tolist(), [(int(c), config['class_names'].get(str(c), 'unknown')) for c in classes]

def extract_scores(bscs, config):
    if config['det_type'] in ["YOLO_v9", "YOLO_v8"]:
        return np.max(bscs[:, 4:], axis=1)
    else:
        return bscs[:, 4]

def extract_classes(bscs, config):
    if config['det_type'] in ["YOLO_v9", "YOLO_v8"]:
        return np.argmax(bscs[:, 4:], axis=1)
    else:
        return np.argmax(bscs[:, 5:], axis=1)

def compute_centers(bscs):
    return np.array([(bscs[:,0] + bscs[:,2]) / 2, (bscs[:,1] + bscs[:,3]) / 2]).T

def compute_areas(bscs):
    return np.array([(bscs[:,2] - bscs[:,0]) * (bscs[:,3] - bscs[:,1])]).flatten()

def sort_by_area(bscs, reverse=False):
    areas = compute_areas(bscs)
    if reverse:
        areas *= -1
    return bscs[np.argsort(areas)]

def non_max_suppression_fast(bscs: np.ndarray, config: dict) -> List:
    """Apply non-maximum suppression to bounding boxes

    Based on:
    https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py

    Parameters
    ----------
    bscs : np.ndarray
        Output array containing bounding boxes in (x1,y1,x2,y2) format and scores

    Returns
    -------
    np.ndarray
        Bounding boxes to keep
    """
    # If no bounding boxes, return empty list
    if len(bscs) == 0:
        return []

    scores = extract_scores(bscs, config)

    # coordinates of bounding boxes
    start_x = bscs[:, 0]
    start_y = bscs[:, 1]
    end_x = bscs[:, 2]
    end_y = bscs[:, 3]

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

        left = np.where(ratio < config['det_iou_thresh'])
        order = order[left]

    return bscs[pick_ids]


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


def non_max_kdtree(bscs: np.ndarray, iou_threshold: float) -> np.ndarray:
    """ Remove overlapping bounding boxes using kdtree

    :param bscs: array of bounding boxes in (xyxy format)
    :param iou_threshold: Threshold for intersection over union
    :return: Filtered output
    """

    centers = compute_centers(bscs)
    bboxes = bscs[:, :4]
    areas = compute_areas(bscs)

    kdtree = cKDTree(centers)
    pick_ids = set()
    removed_ids = set()

    for i, out in enumerate(bscs):
        if i in removed_ids:
            continue
        
        indices = kdtree.query(centers[i], k=min(10, len(bscs)))

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

    return bscs[np.asarray(list(pick_ids))]


def execute_detection(images, session, config):
    images = preprocess(images)

    outs = session.run(None, {config['input_name']: images})
    if config['det_type'] == 'retinanet':
        stacked = np.hstack((outs[0], outs[1][:,np.newaxis], outs[2][:,np.newaxis]))
        out = stacked[np.newaxis, :, :]
    else:
        out = outs[0]

    return postprocess(out, config)


def bscs_to_geojson(raster, bscs, config):
    bboxes, scores, classes = extract_bsc(bscs, config)
    if not len(bboxes):
        return json.dumps({
            "type": "FeatureCollection",
            "features": []
        }, indent=2)
    
    rast_coords = [[
        (b[0], b[1]), # TL
        (b[2], b[1]), # TR
        (b[2], b[3]), # BR
        (b[0], b[3]), # BL
    ] for b in bboxes]
    spatial_coords = [raster.xy(y, x) for c in rast_coords for x,y in c]
    xs, ys = zip(*spatial_coords)
    tx, ty = rasterio.warp.transform(raster.crs, "EPSG:4326", xs, ys)
    feats = []

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        score = scores[i]
        cls = classes[i]
        cid = i * 4

        feats.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[tx[cid + i], ty[cid + i]] for i in range(4)] + [[tx[cid], ty[cid]]]
                ]
            },
            "properties": {
                "score": score,
                "class": cls[1]
            }
        })

    return json.dumps({
        "type": "FeatureCollection",
        "features": feats
    }, indent=2)
