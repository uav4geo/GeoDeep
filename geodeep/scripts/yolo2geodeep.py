import argparse
import sys
import os
import subprocess
import json

def install(package):
    print(f"{package} missing, we'll try to install it")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from ultralytics import YOLO
except ImportError:
    input("Some dependencies are missing. Press Enter to attempt to install them via pip (or CTRL+C to exit): ")
    install("ultralytics")
    from ultralytics import YOLO
try:
    import onnx
except ImportError:
    install("onnx")
    import onnx

try:
    from onnxsim import simplify
except ImportError:
    install("onnxsim")
    from onnxsim import simplify

    from onnxsim import simplify

from onnxruntime.quantization import quantize_dynamic, QuantType,  quantize_static, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process


def read_yaml_keys(filename, keys):
    if isinstance(keys, str):
        keys = [keys]

    r = {}
    with open(filename, 'r') as f:
        lines = [l for l in f.read().split("\n") if l.strip() != ""]
        for l in lines:
            for key in keys:
                if l.startswith(f"{key}:"):
                    r[key] = l[len(key) + 1:].strip()

    out = [r[key] for key in keys]
    if len(out) != len(keys):
        print(f"Cannot read all keys {keys} from {filename}")
        exit(1)
    
    return out[0] if len(out) == 1 else out

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO models to ONNX for use in GeoDeep")
    parser.add_argument(
        "input", 
        type=str, 
        help="Path YOLO model weights (.pt)"
    )
    parser.add_argument(
        "resolution", 
        type=int, 
        help="Target resolution (cm/px) of the model. Set this value to the average ground sampling distance of your training data."
    )
    parser.add_argument(
        "--classes", 
        type=str,
        default=None,
        help="Comma separated list of classes of the model (e.g. car,plane)"
    )
    parser.add_argument(
        "--det-type", 
        type=str,
        default=None,
        help="Detector type"
    )
    parser.add_argument(
        "--det-iou-thresh", 
        type=float,
        default=0.3,
        help="Default Intersection Over Union threshold for model. Default: %(default)s"
    )
    parser.add_argument(
        "--det-conf", 
        type=float,
        default=0.3,
        help="Default confidence for model. Default: %(default)s"
    )
    parser.add_argument(
        "--tiles-overlap", 
        type=float,
        default=10,
        help="Default tiles overlap percentage for model. Default: %(default)s"
    )

    args = parser.parse_args()
    model = YOLO(args.input)
    model.export(format='onnx')
    out_model = os.path.splitext(args.input)[0] + ".onnx"
    out_model_optim = os.path.splitext(args.input)[0] + ".optim.onnx"
    out_model_quant = os.path.splitext(args.input)[0] + ".quant.onnx"
    det_type = "YOLO_v7"

    args_yaml = os.path.join(os.path.dirname(args.input), "..", "args.yaml")
    if args.classes is None and os.path.isfile(args_yaml):
        data_yaml, model = read_yaml_keys(args_yaml, ["data", "model"])

        for v in range(5, 11):
            if model.startswith(f"yolov{v}"):
                det_type = f"YOLO_v{v}"
                break
        
        if data_yaml is None:
            print("Cannot find data.yaml")
            exit(1)
        
        data_yaml = data_yaml.replace("\\", "/")
        if not os.path.isfile(data_yaml):
            data_yaml = os.path.join(os.path.dirname(args.input), *([".."] * 4), data_yaml)
            if not os.path.isfile(data_yaml):
                print(f"Cannot read {data_yaml}")
                exit(1)

        names = read_yaml_keys(data_yaml, "names")
        names = names.replace("'", "").replace("[", "").replace("]", "").split(",")
    elif args.classes is not None:
        names = [s for s in args.classes.split(",") if s.strip() != ""]
    else:
        print("You need to specify the classes of the model via --classes")
        exit(1)
    
    params = {
        'model_type': 'Detector',
        'det_iou_thresh': args.det_iou_thresh,
        'det_type': det_type,
        'resolution': args.resolution, 
        'class_names': {str(i): v for i,v in enumerate(names)}, 
        'det_conf': args.det_conf,
        'tiles_overlap': args.tiles_overlap, 
    }

    print(params)

    # Update meta
    m = onnx.load(out_model)
    for k,v in params.items():
        meta = m.metadata_props.add()
        meta.key = k
        meta.value = json.dumps(v)
    onnx.save(m, out_model)

    model_simp, check = simplify(m)
    onnx.save(m, out_model)
    print(f"Wrote {out_model}")

    quant_pre_process(out_model, out_model_optim, skip_symbolic_shape=True)
    quantized_model = quantize_dynamic(out_model_optim, out_model_quant, weight_type=QuantType.QUInt8)
    os.unlink(out_model_optim)

    print(f"Wrote {out_model_quant} <-- Use this with GeoDeep")
    
    
if __name__ == "__main__":
    main()
