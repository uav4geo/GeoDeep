import rasterio
from .slidingwindow import generate_for_size
from .models import get_model_file
from .inference import create_session
import logging
logger = logging.getLogger("geodeep")

def detect(geotiff, model):
    session, config = create_session(get_model_file(model))

    with rasterio.open(geotiff, 'r') as raster:
        if not raster.is_tiled:
            logger.warning(f"{geotiff} is not tiled. I/O performance will be affected. Consider adding tiles.")
        
        width = raster.shape[0]
        height = raster.shape[1]

        windows = generate_for_size(width, height, config['tiles_size'], config['tiles_overlap'] / 100.0)

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

