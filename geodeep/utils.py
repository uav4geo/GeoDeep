import numpy as np
import json

# Originally edited from https://github.com/PUTvision/qgis-plugin-deepness
def xywh2xyxy(x):
    """Convert bounding box from (x,y,w,h) to (x1,y1,x2,y2) format"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

