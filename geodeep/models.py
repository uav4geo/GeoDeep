import os
import urllib.request
import logging
logger = logging.getLogger("geodeep")

MODELS = {
    'cars': 'https://huggingface.co/datasets/UAV4GEO/GeoDeep-Models/resolve/main/car_aerial_detection_yolo7_ITCVD_deepness.onnx',
    # TODO: add more
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

# This can be overriden at runtime    
cache_dir = os.path.join(get_user_cache_dir(), "geodeep")

def list_models():
    return MODELS.keys()

def get_model_file(name):
    if name.startswith("http"):
        url = name
    else:
        url = MODELS.get(name)
    
    if not url:
        raise Exception(f"Invalid model: {name}, not in {list_models()}")
    
    filename = os.path.basename(url)
    model_path = os.path.join(cache_dir, filename)

    if not os.path.isfile(model_path):
        os.makedirs(cache_dir, exist_ok=True)
        logger.warning(f"Downloading {url} to {model_path}")
        urllib.request.urlretrieve(url, model_path)
    
    return os.path.abspath(model_path)