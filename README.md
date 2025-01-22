# GeoDeep

A fast, lightweight Python library for AI object detection in geospatial rasters (GeoTIFFs). 

![Image](https://github.com/user-attachments/assets/9dc7b98b-9233-458b-976e-c619c3a608cf)

## Install

```bash
pip install -U geodeep
```

## Usage

### From the command line

```bash
geodeep [geotiff] [modelID or path to ONNX model]
```

Example:

```bash
geodeep orthophoto.tif cars
```

Here GeoDeep will find cars in the orthophoto and write the result as a GeoJSON file containing the bounding boxes, confidence scores and class labels.

A list of up-to-date modelID can be retrieved via:

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

| **Model**    | **Description**                                                                                                                                            | **Resolution (cm/px)** | **Experimental**   |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------ |
| cars         | YOLOv7-m model for cars detection on aerial images. Based on [ITCVD](https://arxiv.org/pdf/1801.07339).                                                    | 10                     |                    |
| trees        | Retinanet tree crown detection model from [DeepForest](https://deepforest.readthedocs.io/en/v1.5.0/user_guide/02_prebuilt.html#tree-crown-detection-model) | 10                     |                    |
| birds        | Retinanet bird detection model from [DeepForest](https://deepforest.readthedocs.io/en/v1.5.0/user_guide/02_prebuilt.html#bird-detection-model)             | 2                      | :heavy_check_mark: |
| trees_yolov7 | YOLOv9 model for treetops detection on aerial images. Model is trained on a mix of publicly available datasets.                                            | 10                     | :heavy_check_mark: |

All ONNX models are published on https://huggingface.co/datasets/UAV4GEO/GeoDeep-Models

## Creating New Models

Instructions coming soon. The basic idea is to create an ONNX model (see the [retinanet conversion script](https://github.com/uav4geo/GeoDeep/blob/main/geodeep/scripts/convert_retinanet_to_onnx.py)) and possibly make some modifications to GeoDeep to handle different conventions in model architectures via conditional checking.

## Why GeoDeep?

Compared to other software packages (e.g. [Deepness](https://github.com/PUTvision/qgis-plugin-deepness)), GeoDeep relies only on two dependencies, `rasterio` and `onnxruntime`. This makes it simple and lightweight.

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