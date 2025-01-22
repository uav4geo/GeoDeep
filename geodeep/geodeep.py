import rasterio
import numpy as np
from .slidingwindow import generate_for_size
from .models import get_model_file
from .inference import create_session
from .detection import execute, non_max_suppression_fast, extract_bsc, non_max_kdtree, sort_by_area, to_geojson
import logging
logger = logging.getLogger("geodeep")


def detect(geotiff, model, output_type='bsc', progress_callback=None):
    """
    Perform object detection on a GeoTIFF
    """
    current_progress = 0
    def p(text, perc=0):
        nonlocal current_progress
        current_progress += perc
        if progress_callback is not None:
            progress_callback(text, current_progress)
    
    p("Loading model")
    session, config = create_session(get_model_file(model, progress_callback))
    p("Model loaded", 5)

    with rasterio.open(geotiff, 'r') as raster:
        if not raster.is_tiled:
            logger.warning(f"{geotiff} is not tiled. I/O performance will be affected. Consider adding tiles.")
        
        # cm/px
        input_res = round(max(abs(raster.transform[0]), abs(raster.transform[4])), 4) * 100
        model_res = config['resolution']
        scale_factor = 1

        if input_res < model_res:
            scale_factor = int(model_res // input_res)
        
        height = raster.shape[0]
        width = raster.shape[1]

        windows = generate_for_size(width, height, config['tiles_size'] * scale_factor, config['tiles_overlap'] / 100.0)
        outputs = []

        # Skip alpha
        indexes = raster.indexes
        if len(indexes) > 1 and raster.colorinterp[-1] == rasterio.enums.ColorInterp.alpha:
            indexes = indexes[:-1]

        num_wins = len(windows)
        progress_per_win = 90 / num_wins if num_wins > 0 else 0
        
        for idx, w in enumerate(windows):
            p(f"Processing tile {idx}/{num_wins}", progress_per_win)
            img = raster.read(indexes=indexes, window=w, boundless=True, fill_value=0, out_shape=(
                len(indexes),
                config['tiles_size'],
                config['tiles_size'],
            ), resampling=rasterio.enums.Resampling.nearest)
            
            res = execute(img, session, config)

            # from .debug import draw_boxes, save_raster
            # save_raster(img, f"tmp/tile_{idx}.tif", raster)

            if len(res):
                # bboxes, scores, classes = extract_bsc(res, config)
                # save_raster(img, f"tmp/tile_{idx}.tif", raster)
                # draw_boxes(f"tmp/tile_{idx}.tif", f"tmp/tile_{idx}_out.tif", bboxes, scores)
                
                # Scale/shift bbox coordinates from tile space to raster space
                res[:,0:4] = res[:,0:4] * scale_factor + np.array([w.col_off, w.row_off, w.col_off, w.row_off])
                outputs.append(res)
        
        p("Finalizing", 5)

        outputs = np.vstack(outputs)
        outputs = non_max_suppression_fast(outputs, config['det_iou_thresh'])
        outputs = sort_by_area(outputs, reverse=True)
        outputs = non_max_kdtree(outputs, config['det_iou_thresh'])

        if output_type == 'raw':
            return outputs
        elif output_type == 'bsc':
            bboxes, scores, classes = extract_bsc(outputs, config)
            # from .debug import draw_boxes
            # draw_boxes(geotiff, "tmp/out.tif", bboxes, scores)
            return bboxes, scores, classes
        elif output_type == 'geojson':
            return to_geojson(raster, outputs, config)

