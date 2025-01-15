import rasterio
from .slidingwindow import generate_for_size
from .models import get_model_file
import logging
logger = logging.getLogger("geodeep")

def detect(geotiff, model, patch_size=400, patch_overlap=0.05, iou_threshold=0.15):
    model_file = get_model_file(model)
    print(model_file)
    return

    with rasterio.open(geotiff) as raster:
        if not raster.is_tiled:
            logger.warning(f"{geotiff} is not tiled. I/O performance will be affected. Consider adding tiles.")
        
        width = raster.shape[0]
        height = raster.shape[1]

        windows = generate_for_size(width, height, patch_size, patch_overlap)

        for idx, w in enumerate(windows):
            data = raster.read(window=w)

            # TODO TEST
            profile = raster.profile
            profile.update({
                "width": w.width,
                "height": w.height,
                "transform": raster.window_transform(w)
            })

            with rasterio.open(geotiff + f"{idx}.tif", "w", **profile) as dst:
                dst.write(data)
                print(f"Wrote {idx}")

    return []

