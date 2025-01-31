import rasterio
import numpy as np
from .detection import preprocess

def postprocess(model_output, config):
    mask = np.argmax(model_output, axis=1).astype(np.float32)
    return mask

def execute_segmentation(images, session, config):
    images = preprocess(images)

    outs = session.run(None, {config['input_name']: images})
    out = outs[0]

    return postprocess(out, config)


