import rasterio
import numpy as np
from .slidingwindow import generate_for_size
from .models import get_model_file
from .inference import create_session
from .utils import estimate_raster_resolution, cls_names_map
from .detection import execute_detection, non_max_suppression_fast, extract_bsc, non_max_kdtree, sort_by_area, to_geojson
from .segmentation import execute_segmentation
import logging
logger = logging.getLogger("geodeep")


def detect(geotiff, model, output_type='bsc', 
            conf_threshold=None, resolution=None, classes=None, 
            max_threads=None, progress_callback=None):
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
    session, config = create_session(get_model_file(model, progress_callback), max_threads=max_threads)
    p("Model loaded", 5)

    # Override defaults if needed
    if conf_threshold is not None:
        config['det_conf'] = conf_threshold
    
    if resolution is not None:
        config['resolution'] = resolution
    
    if classes is not None:
        cn_map = cls_names_map(config['class_names'])
        config['det_classes'] = [cn_map[cls_name] for cls_name in cn_map if cls_name in classes]
        
    with rasterio.open(geotiff, 'r') as raster:
        if not raster.is_tiled:
            logger.warning(f"\n{geotiff} is not tiled. I/O performance will be affected. Consider adding tiles.")
        
        # cm/px
        input_res = round(max(abs(raster.transform[0]), abs(raster.transform[4])), 4) * 100
        if input_res <= 0:
            input_res = estimate_raster_resolution(raster)
            logger.warning(f"\n{geotiff} does not seem to have a valid transform, estimated raster resolution: {input_res} cm/px")
        
        model_res = config['resolution']
        scale_factor = 1

        if input_res < model_res:
            scale_factor = int(model_res // input_res)
        
        height = raster.shape[0]
        width = raster.shape[1]

        windows = generate_for_size(width, height, config['tiles_size'] * scale_factor, config['tiles_overlap'] / 100.0, clip=False)
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
            ), resampling=rasterio.enums.Resampling.bilinear)

            if config['model_type'] == 'Detector':
                res = execute_detection(img, session, config)
            elif config['model_type'] == 'Segmentor':
                res = execute_segmentation(img, session, config)
            
            # elif config['model_type'] == 
            # from .debug import draw_boxes, save_raster
            # save_raster(img, f"tmp/tiles/tile_{idx}.tif", raster)

            if len(res):
                # bboxes, scores, classes = extract_bsc(res, config)
                # save_raster(img, f"tmp/tiles/tile_{idx}.tif", raster)
                # draw_boxes(f"tmp/tiles/tile_{idx}.tif", f"tmp/tiles/tile_{idx}_out.tif", bboxes, scores)
                
                # Scale/shift bbox coordinates from tile space to raster space
                res[:,0:4] = res[:,0:4] * scale_factor + np.array([w.col_off, w.row_off, w.col_off, w.row_off])
                outputs.append(res)
        
        p("Finalizing", 5)
        if len(outputs):
            outputs = np.vstack(outputs)
            outputs = non_max_suppression_fast(outputs, config)
            outputs = sort_by_area(outputs, reverse=True)
            outputs = non_max_kdtree(outputs, config['det_iou_thresh'])
        else:
            outputs = np.array([])
        
        if output_type == 'raw':
            return outputs
        elif output_type == 'bsc':
            bboxes, scores, classes = extract_bsc(outputs, config)
            # from .debug import draw_boxes
            # draw_boxes(geotiff, "tmp/out.tif", bboxes, scores)
            return bboxes, scores, classes
        elif output_type == 'geojson':
            return to_geojson(raster, outputs, config)

