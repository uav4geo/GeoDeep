import onnxruntime as ort
import warnings
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

def create_session(model_file):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]

    session = ort.InferenceSession(model_file, options=options, providers=providers)
    inputs = session.get_inputs()
    if len(inputs) > 1:
        raise Exception("ONNX model: unsupported number of inputs")
    
    meta = session.get_modelmeta().custom_metadata_map
    config = {
        'det_type': meta.get('det_type', 'YOLO_v5_or_v7_default'),
        'det_conf': float(meta.get('det_conf', 0.3)),
        'det_iou_thresh': float(meta.get('det_iou_thresh', 0.8)),
        'resolution': float(meta.get('resolution', 10)),
        'class_names': meta.get('class_names', {}),
        'model_type': meta.get('model_type', 'Detector'),
        'tiles_overlap': float(meta.get('tiles_overlap', 5)), # percentage
        'tiles_size': inputs[0].shape[-1],
        'input_shape': inputs[0].shape,
        'input_name': inputs[0].name,
    }

    return session, config