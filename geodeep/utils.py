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

def simple_progress(text, perc):
    bar_chars = 20
    f = perc / 100
    fill = int(bar_chars * f)
    empty = bar_chars - fill
    bar = f"{'█' * fill}{'-' * empty}"
    print(f"\r\033[K[{bar}] {perc:.1f}% {text}", end='', flush=True)
    if perc == 100:
        print("\r\033[K")