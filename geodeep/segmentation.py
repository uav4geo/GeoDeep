import rasterio
import numpy as np
from .detection import preprocess

def postprocess(model_output, config):
    mask = np.argmax(model_output, axis=1).astype(np.float32)
    with rasterio.open("tmp/mask1.tif", "w", dtype=np.float32, count=1, width=256, height=256) as dst:
        dst.write(mask)
        # dst.write(mask)
        print(f"Wrote mask")
    exit(1)

def execute_segmentation(images, session, config):
    images = preprocess(images)

    outs = session.run(None, {config['input_name']: images})
    out = outs[0]

    return postprocess(out, config)


