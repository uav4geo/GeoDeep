import rasterio
import numpy as np
from .slidingwindow import generate_for_size
from .models import get_model_file
from .inference import create_session
from .utils import estimate_raster_resolution, cls_names_map
from .detection import execute_detection, non_max_suppression_fast, extract_bsc, non_max_kdtree, sort_by_area, to_geojson
from .segmentation import execute_segmentation
from .utils import rect_intersect
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
    
    detector = config['model_type'] == 'Detector'
    segmentor = config['model_type'] == 'Segmentor'
        
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
        scaled_h = height // scale_factor
        scaled_w = width // scale_factor
        tiles_overlap = 0.3 #config['tiles_overlap'] / 100.0

        windows = generate_for_size(width, height, config['tiles_size'] * scale_factor, tiles_overlap, clip=False)
        bscs = []
        segmask = None

        if segmentor:
            segmask = np.zeros((scaled_h, scaled_w), dtype=np.uint8)

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

            # from .debug import draw_boxes, save_raster
            # save_raster(img, f"tmp/tiles/tile_{idx}.tif", raster)

            if detector:
                bsc = execute_detection(img, session, config)

                if len(bsc):
                    # bboxes, scores, classes = extract_bsc(bsc, config)
                    # save_raster(img, f"tmp/tiles/tile_{idx}.tif", raster)
                    # draw_boxes(f"tmp/tiles/tile_{idx}.tif", f"tmp/tiles/tile_{idx}_out.tif", bboxes, scores)
                    
                    # Scale/shift bbox coordinates from tile space to raster space
                    bsc[:,0:4] = bsc[:,0:4] * scale_factor + np.array([w.col_off, w.row_off, w.col_off, w.row_off])
                    bscs.append(bsc)
            elif segmentor:
                mask = execute_segmentation(img, session, config)

                row_off = int(np.round(w.row_off / scale_factor))
                col_off = int(np.round(w.col_off / scale_factor))
                tile_w, tile_h = mask.shape[1:]

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

                mask = mask[:,pad_t:pad_t+tile_h,pad_l:pad_l+tile_w]

                tr, sr = rect_intersect((col_off, row_off, tile_w, tile_h), (0, 0, scaled_w, scaled_h))
                if tr is not None and sr is not None:
                    segmask[sr[1]:sr[1]+sr[3], sr[0]:sr[0]+sr[2]] = mask[:, tr[1]:tr[1]+tr[3], tr[0]:tr[0]+tr[2]]
                    segmask[sr[1]:sr[1]+sr[3], sr[0]:sr[0]+sr[2]] *= (idx + 1)
                else:
                    logger.warning(f"Cannot merge segmentation tile {idx}")
        
        p("Finalizing", 5)

        if detector:
            if len(bscs):
                bscs = np.vstack(bscs)
                bscs = non_max_suppression_fast(bscs, config)
                bscs = sort_by_area(bscs, reverse=True)
                bscs = non_max_kdtree(bscs, config['det_iou_thresh'])
            else:
                bscs = np.array([])
            
            if output_type == 'raw':
                return bscs
            elif output_type == 'bsc':
                bboxes, scores, classes = extract_bsc(bscs, config)
                # from .debug import draw_boxes
                # draw_boxes(geotiff, "tmp/out.tif", bboxes, scores)
                return bboxes, scores, classes
            elif output_type == 'geojson':
                return to_geojson(raster, bscs, config)
        elif segmentor:
            with rasterio.open("tmp/mask1.tif", "w", dtype=np.uint8, count=1, width=scaled_w, height=scaled_h) as dst:
                print(segmask.shape)
                dst.write(segmask, 1)
                # dst.write(mask)
                print(f"Wrote mask")
            exit(1)

            return mask

