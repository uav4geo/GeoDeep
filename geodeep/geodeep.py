import rasterio
import numpy as np
from .slidingwindow import generate_for_size
from .models import get_model_file
from .inference import create_session
from .detection import execute, non_max_suppression_fast, extract_bsc, non_max_kdtree, sort_by_area
import logging
logger = logging.getLogger("geodeep")

def detect(geotiff, model):
    session, config = create_session(get_model_file(model))

    with rasterio.open(geotiff, 'r') as raster:
        if not raster.is_tiled:
            logger.warning(f"{geotiff} is not tiled. I/O performance will be affected. Consider adding tiles.")
        
        height = raster.shape[0]
        width = raster.shape[1]

        windows = generate_for_size(width, height, config['tiles_size'], config['tiles_overlap'] / 100.0)
        outputs = []

        for idx, w in enumerate(windows):
            img = raster.read(window=w, boundless=True, fill_value=0)
            res = execute(img, session, config)

            from .debug import draw_boxes, save_raster
            save_raster(img, f"tmp/tile_{idx}.tif", raster)

            if len(res):
                bboxes, scores, classes = extract_bsc(res)
                save_raster(img, f"tmp/tile_{idx}.tif", raster)
                draw_boxes(f"tmp/tile_{idx}.tif", f"tmp/tile_{idx}_out.tif", bboxes, scores)

                # Shift bbox coordinates from tile space to raster space
                res[:,0:4] += np.array([w.col_off, w.row_off, w.col_off, w.row_off])
                outputs.append(res)

        outputs = np.vstack(outputs)
        outputs = non_max_suppression_fast(outputs, config['det_iou_thresh'])
        outputs = sort_by_area(outputs, reverse=True)
        outputs = non_max_kdtree(outputs, config['det_iou_thresh'])

        bboxes, scores, classes = extract_bsc(outputs)

        from .debug import draw_boxes
        draw_boxes(geotiff, "tmp/out.tif", bboxes, scores)

    return []

