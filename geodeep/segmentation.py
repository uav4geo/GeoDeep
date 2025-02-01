import rasterio
import rasterio.features
import json
import numpy as np
from .detection import preprocess
import logging
logger = logging.getLogger("geodeep")

def postprocess(model_output, config):
    mask = np.argmax(model_output, axis=1).astype(np.float32)
    return mask

def execute_segmentation(images, session, config):
    images = preprocess(images)

    outs = session.run(None, {config['input_name']: images})
    out = outs[0]

    return postprocess(out, config)


def rect_intersect(rect1, rect2):
    """
    Given two rectangles, compute the intersection rectangle and return 
    its coordinates in the coordinate system of both rectangles.
    
    Each rectangle is represented as (x, y, width, height).

    Returns:
    - (r1_x, r1_y, iw, ih): Intersection in rect1's local coordinates
    - (r2_x, r2_y, iw, ih): Intersection in rect2's local coordinates
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    ix = max(x1, x2)  # Left boundary
    iy = max(y1, y2)  # Top boundary
    ix2 = min(x1 + w1, x2 + w2)  # Right boundary
    iy2 = min(y1 + h1, y2 + h2)  # Bottom boundary

    # Compute intersection
    iw = max(0, ix2 - ix)
    ih = max(0, iy2 - iy)

    # If no intersection
    if iw == 0 or ih == 0:
        return None, None

    # Compute local coordinates
    r1_x = ix - x1
    r1_y = iy - y1
    r2_x = ix - x2
    r2_y = iy - y2

    return (r1_x, r1_y, iw, ih), (r2_x, r2_y, iw, ih)


def merge_mask(tile_mask, mask, window, width, height, tiles_overlap=0, scale_factor = 1.0):
    w = window
    row_off = int(np.round(w.row_off / scale_factor))
    col_off = int(np.round(w.col_off / scale_factor))
    tile_w, tile_h = tile_mask.shape[1:]

    pad_x = int(tiles_overlap * tile_w) // 2
    pad_y = int(tiles_overlap * tile_h) // 2

    pad_l = 0
    pad_r = 0
    pad_t = 0
    pad_b = 0

    if w.col_off > 0:
        pad_l = pad_x
    if w.col_off + w.width < width:
        pad_r = pad_x
    if w.row_off > 0:
        pad_t = pad_y
    if w.row_off + w.height < height:
        pad_b = pad_y
    
    row_off += pad_t
    col_off += pad_l
    tile_w -= pad_l + pad_r
    tile_h -= pad_t + pad_b

    tile_mask = tile_mask[:,pad_t:pad_t+tile_h,pad_l:pad_l+tile_w]

    tr, sr = rect_intersect((col_off, row_off, tile_w, tile_h), (0, 0, mask.shape[1], mask.shape[0]))
    if tr is not None and sr is not None:
        mask[sr[1]:sr[1]+sr[3], sr[0]:sr[0]+sr[2]] = tile_mask[:, tr[1]:tr[1]+tr[3], tr[0]:tr[0]+tr[2]]
        # mask[sr[1]:sr[1]+sr[3], sr[0]:sr[0]+sr[2]] *= (idx + 1)
    else:
        logger.warning(f"Cannot merge segmentation tile {w}")
        

def mask_to_geojson(raster, mask, config, scale_factor=1.0):
    transform = list(raster.transform)
    transform[0] *= scale_factor
    transform[4] *= scale_factor

    shapes = rasterio.features.shapes(mask, transform=transform)

    # TODO: map classes, names
    # TODO: remove speckles (median filter? dilation/erosion?)
    
    return json.dumps({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {"value": value}
            }
            for geom, value in shapes
        ]
    }, indent=2)
