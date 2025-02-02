import numpy as np
import json
import math

# Originally edited from https://github.com/PUTvision/qgis-plugin-deepness
def xywh2xyxy(x):
    """Convert bounding box from (x,y,w,h) to (x1,y1,x2,y2) format"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def simple_progress(text, perc):
    bar_chars = 20
    f = perc / 100
    fill = int(bar_chars * f)
    empty = bar_chars - fill
    bar = f"{'█' * fill}{'-' * empty}"
    print(f"\r\033[K[{bar}] {perc:.1f}% {text}", end='', flush=True)
    if perc == 100:
        print("\r\033[K")

def estimate_raster_resolution(raster):
    if raster.crs is None:
        return 10 # Wild guess cm/px
    
    bounds = raster.bounds
    width = raster.width
    height = raster.height
    crs = raster.crs
    res_x = (bounds.right - bounds.left) / width
    res_y = (bounds.top - bounds.bottom) / height

    if crs.is_geographic:
        center_lat = (bounds.top + bounds.bottom) / 2
        earth_radius = 6378137.0
        meters_lon = math.pi / 180 * earth_radius * math.cos(math.radians(center_lat))
        meters_lat = math.pi / 180 * earth_radius
        res_x *= meters_lon
        res_y *= meters_lat

    return round(max(abs(res_x), abs(res_y)), 4) * 100 # cm/px

def cls_names_map(class_names):
    # {"0": "tree"} --> {"tree": 0}
    d = {}
    for i in class_names:
        d[class_names[i]] = int(i)
    return d

try:
    from scipy.ndimage import median_filter
except ImportError:
    def median_filter(arr, size=5):
        assert size % 2 == 1, "Kernel size must be an odd number."
        if arr.shape[0] <= size or arr.shape[1] <= size:
            return arr
        
        pad_size = size // 2
        padded = np.pad(arr, pad_size, mode='edge')
        shape = (arr.shape[0], arr.shape[1], size, size)
        strides = padded.strides + padded.strides
        view = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        return np.median(view, axis=(2, 3)).astype(arr.dtype)