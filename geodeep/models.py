import os
import urllib.request
import time

MODELS = {
    'cars': 'https://huggingface.co/datasets/UAV4GEO/GeoDeep-Models/resolve/main/car_aerial_detection_yolo7_ITCVD_deepness.onnx',
    'trees': 'https://huggingface.co/datasets/UAV4GEO/GeoDeep-Models/resolve/main/tree_crown_detection_retinanet_deepforest.onnx',

    # Experimental
    'trees_yolov9': 'https://huggingface.co/datasets/UAV4GEO/GeoDeep-Models/resolve/main/yolov9_trees.onnx',
    'birds': 'https://huggingface.co/datasets/UAV4GEO/GeoDeep-Models/resolve/main/bird_detection_retinanet_deepforest.onnx',

    # TODO add more
}

def get_user_cache_dir():
    if os.name == 'nt':  # Windows
        return os.path.join(os.getenv('LOCALAPPDATA', os.path.expanduser('~')), 'Cache')
    elif os.name == 'posix':  # Linux or macOS
        if 'darwin' in os.sys.platform:  # macOS
            return os.path.expanduser('~/Library/Caches')
        else:  # Linux
            return os.path.expanduser('~/.cache')
    else:
        return ""

# This can be overridden at runtime    
cache_dir = os.path.join(get_user_cache_dir(), "geodeep")

def list_models():
    return list(MODELS.keys())

def get_model_file(name, progress_callback=None):
    if name.startswith("http"):
        url = name
    else:
        url = MODELS.get(name)
        if not url and not os.path.isfile(name):
            raise Exception(f"Invalid model: {name}, not in {list_models()}")

        if url is None and os.path.isfile(name):
            return name
    
    try:
        filename = os.path.basename(url)
        model_path = os.path.join(cache_dir, filename)
        last_update = 0

        def progress(block_num, block_size, total_size):
            nonlocal last_update
            now = time.time()
            if progress_callback is not None and total_size > 0 and now - last_update >= 1:
                progress_callback(f"Downloading model", block_num * block_size / total_size * 5)
                last_update = now

        if not os.path.isfile(model_path):
            os.makedirs(cache_dir, exist_ok=True)
            urllib.request.urlretrieve(url, model_path, progress)
        
        return os.path.abspath(model_path)
    except Exception as e:
        # Cleanup possibly corrupted file
        if os.path.isfile(model_path):
            os.unlink(model_path)
        raise e