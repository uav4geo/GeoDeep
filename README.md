# GeoDeep

A fast, easy to use, lightweight Python library for AI object detection in geospatial rasters (GeoTIFFs), with pre-built models included.

![Image](https://github.com/user-attachments/assets/9dc7b98b-9233-458b-976e-c619c3a608cf)

## Install

```bash
pip install -U geodeep
```

## Usage

### From the command line

```bash
geodeep [geotiff] [model ID or path to ONNX model]
```

Example:

```bash
geodeep orthophoto.tif cars
```

Here GeoDeep will find cars in the orthophoto and write the result as a GeoJSON file containing the bounding boxes, confidence scores and class labels.

A list of up-to-date model IDs can be retrieved via:

```bash
geodeep --list-models
```

See also `geodeep --help`.

### From Python

```python
from geodeep import detect
bboxes, scores, classes = detect('orthophoto.tif', 'cars')
print(bboxes) # <-- [[x_min, y_min, x_max, y_max], [...]]
print(scores) # <-- [score, ...]
print(classes) # <-- [(id: int, label: str), ...]

geojson = detect('orthophoto.tif', 'cars', output_type="geojson")
```

Models by default will be cached in `~/.cache/geodeep`. You can change that with:

```python
from geodeep import models
models.cache_dir = "your/cache/path"
```

## Models

| **Model**    | **Description**                                                                                                                                                                          | **Resolution (cm/px)** | **Experimental**   | **Classes** |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------ | ----------- |
| cars         | YOLOv7-m model for cars detection on aerial images. Based on [ITCVD](https://arxiv.org/pdf/1801.07339).                                                                                  | 10                     |                    | car         |
| trees        | Retinanet tree crown detection model from [DeepForest](https://deepforest.readthedocs.io/en/v1.5.0/user_guide/02_prebuilt.html#tree-crown-detection-model)                               | 10                     | :heavy_check_mark: | tree        |
| trees_yolov9 | YOLOv9 model for treetops detection on aerial images. Model is trained on a mix of publicly available datasets.                                                                          | 10                     | :heavy_check_mark: | tree        |
| birds        | Retinanet bird detection model from [DeepForest](https://deepforest.readthedocs.io/en/v1.5.0/user_guide/02_prebuilt.html#bird-detection-model)                                           | 2                      | :heavy_check_mark: | bird        |
| planes       | YOLOv7 tiny model for object detection on satellite images. Based on the [Airbus Aircraft Detection dataset](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset). | 70                     | :heavy_check_mark: | plane       |
| aerovision   | YOLOv8-m for multi-class detection on aerial images.                                                                                                                                     | 30                     | :heavy_check_mark: | [1]         |

1. small-vehicle, large-vehicle,plane,storage-tank,ship,dock,ground-track-field,soccer-field,tennis-court,swimming-pool,baseball-field,road-circle,basketball-court,bridge,helicopter,crane

All ONNX models are published on https://huggingface.co/datasets/UAV4GEO/GeoDeep-Models

## Training New Models

In short, first you need to train a [YOLO](https://en.wikipedia.org/wiki/You_Only_Look_Once) model, then you run `yolo2geodeep`. See below for details.

### Requirements

You need a decent GPU and plenty of RAM. It's possible to train models on a CPU, but it will take weeks (maybe even months). There's also [platforms](https://roboflow.com/train) that will do the training for you if you don't have the necessary hardware.

### Step 1. Gather annotated images

A good point to start is https://universe.roboflow.com/browse/aerial, but the quality of the datasets is all over the place. Always inspect before using. When downloading a dataset, choose the YOLOv8 format.

You can also [annotate](https://roboflow.com/annotate) your own images.

Aim to gather at least 1000 training images for decent results.

### Step 2. Train a YOLO model

For up to date instructions, follow the steps on https://docs.ultralytics.com/modes/train/. Alsk make sure to install a GPU version of pytorch (https://pytorch.org/get-started/locally/).

Once you have a folder with your annotated images (e.g. `dataset/train`, `dataset/valid`), check your `data.yaml` to make sure you have the correct number of classes, then run:

`yolo train task=detect model=yolov8s.pt data=dataset\data.yaml epochs=400`

There's also several [settings](https://docs.ultralytics.com/usage/cfg/) you can tweak, but start with the defaults.

Once the processes is done, you'll end up with a `best.pt` (model weights) file, usually in `runs/detect/trainX/weights/best.pt`.

### Step 3. Convert the YOLO model to ONNX

Before converting, you should estimate the ground sampling distance (GSD) resolution of your training data (in cm/px). This affects the model quality quite a bit so it's important to have a good estimate. If you're unsure, you can just start with a reasonable value (e.g. 10 or 20 for aerial datasets) and run a few experiments to see which value yields the best results.

Then:

```bash
yolo2geodeep runs/detect/trainX/weights/best.pt [resolution]

[...]
Wrote runs/detect/trainX/weights/best.quant.onnx <-- Use this with GeoDeep
```

You can finally run:

```bash
geodeep orthophoto.tif runs/detect/trainX/weights/best.quant.onnx
```

You can also convert existing ONNX models for use with GeoDeep. See the [retinanet conversion script](https://github.com/uav4geo/GeoDeep/blob/main/geodeep/scripts/convert_retinanet_to_onnx.py) for an example. In some cases modifications to GeoDeep might be required if the model architecture is not supported. Currently GeoDeep supports:

 * YOLO 5,6,7,8,9
 * Retinanet

Other architectures can be added. Pull requests welcome!

### Step 4. (Optional) Share Your Model

The most convenient way to deploy your model is to share it. Open a pull request on https://huggingface.co/datasets/UAV4GEO/GeoDeep-Models/tree/main and we'll include it in GeoDeep!

## Inspect Models

You can inspect an existing model by running:

```bash
geodeep-inspect [model ID or path to ONNX model]
```

For example:

```bash
geodeep-inspect cars

det_type: YOLO_v5_or_v7_default
det_conf: 0.3
det_iou_thresh: 0.8
det_classes: []
resolution: 10.0
class_names: {'0': 'car'}
model_type: Detector
tiles_overlap: 10.0
tiles_size: 640
input_shape: [1, 3, 640, 640]
input_name: images
```

## Why GeoDeep?

Compared to other software packages (e.g. [Deepness](https://github.com/PUTvision/qgis-plugin-deepness)), GeoDeep relies only on two dependencies, `rasterio` and `onnxruntime`. This makes it simple and lightweight.

## Does this need a GPU?

It does not! Models are tuned to run fast on the CPU.

## Contributing

We welcome contributions! Pull requests are welcome.

## Roadmap Ideas

 - [ ] Train more detection models
 - [ ] Add support for semantic segmentation models
 - [ ] Faster inference optimizations

## Support the Project

There are many ways to contribute to the project:

 - ⭐️ us on GitHub.
 - Help us test the application.
 - Become a contributor!

## Credits

GeoDeep was inspired and uses some code from [Deepness](https://github.com/PUTvision/qgis-plugin-deepness) and [DeepForest](https://github.dev/weecology/DeepForest).

 ## License

The code in this repository is licensed under the AGPLv3.

Made with ❤️ by UAV4GEO
