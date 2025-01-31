# Script to convert some deepforest model checkpoints to ONNX

import torch
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights
import numpy as np
from PIL import Image
from onnxruntime.quantization import quantize_dynamic, QuantType,  quantize_static, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxsim import simplify
import onnx


INPUT_PT = "bird.pt"
TEST_IMAGE = "AWPE Pigeon Lake 2020 DJI_0005.JPG"
CLASS_NAMES = {"0": "bird"}
RESOLUTION = 2 # cm/px

p, ext = os.path.splitext(INPUT_PT)
OUT_MODEL = f"{p}.onnx"
OUT_MODEL_OPTIM = f"{p}.optim.onnx"
OUT_MODEL_QUANT = f"{p}.quant.onnx"

resnet = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
model = RetinaNet(backbone=resnet.backbone, num_classes=1)
model.nms_thresh = 0.05
model.score_thresh = 0.1

chk = torch.load(INPUT_PT)
model.load_state_dict(chk)
model.eval() 

img = np.array(Image.open(TEST_IMAGE).convert("RGB")).astype("float32").transpose((2, 0, 1))

image = torch.tensor(img)
image = image / 255
image = image.unsqueeze(0)

with torch.no_grad():
    prediction = model(image)



torch.onnx.export(model,         # model being run 
            image,       # model input (or a tuple for multiple inputs) 
            OUT_MODEL,       # where to save the model  
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=11,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['images'],   # the model's input names 
            output_names = ['boxes', 'scores', 'labels'], # the model's output names 
        )


# Add meta
params = {
    'model_type': 'Detector',
    'det_iou_thresh': 0.4,
    'det_type': 'retinanet',
    'resolution': RESOLUTION, 
    'class_names': CLASS_NAMES, 
    'det_conf': 0.3, 
    'tiles_overlap': 5, 
}

m = onnx.load(OUT_MODEL)
for k,v in params.items():
    meta = m.metadata_props.add()
    meta.key = k
    meta.value = json.dumps(v)

model_simp, check = simplify(m)
onnx.save(m, OUT_MODEL)

quant_pre_process(OUT_MODEL, OUT_MODEL_OPTIM, skip_symbolic_shape=True)
quantized_model = quantize_dynamic(OUT_MODEL_OPTIM, OUT_MODEL_QUANT, weight_type=QuantType.QUInt8)

print(f"Wrote {OUT_MODEL_QUANT}")


from geodeep.inference import create_session
from geodeep.detection import extract_bsc, execute_detection
from geodeep.debug import draw_boxes

session, config = create_session(OUT_MODEL_QUANT)
res = execute_detection(img, session, config)
bboxes, scores, classes = extract_bsc(res, config)

draw_boxes(TEST_IMAGE, "out.tif", bboxes, scores)


