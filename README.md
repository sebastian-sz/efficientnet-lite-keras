EfficientNet Lite models adapted to Keras functional API.

### Changelog:
* Nov 2021:
  * Added separate `get_preprocessing_layer` utility function.

# Table of contents
1. [Introduction](https://github.com/sebastian-sz/efficientnet-lite-keras#introduction)
2. [Quickstart](https://github.com/sebastian-sz/efficientnet-lite-keras#quickstart)
3. [Installation](https://github.com/sebastian-sz/efficientnet-lite-keras#installation)
4. [How to use](https://github.com/sebastian-sz/efficientnet-lite-keras#how-to-use)
5. [Original Weights](https://github.com/sebastian-sz/efficientnet-lite-keras#original-weights)

# Introduction
This is a package with EfficientNet-Lite model variants adapted to Keras.  

EfficientNet-Lite variants are modified versions of EfficientNet models, better suited for mobile and embedded devices.   

The model's weights are converted from [original repository](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/).

# Quickstart
The design was meant to mimic the usage of `keras.applications`:
```python
# Install
!pip install git+https://github.com/sebastian-sz/efficientnet-lite-keras@main

# Import package:
from efficientnet_lite import EfficientNetLiteB0
import tensorflow as tf

# Use model directly:
model = EfficientNetLiteB0(weights='imagenet', input_shape=(224, 224, 3))
model.summary()

# Or to extract features / fine tune:
backbone = EfficientNetLiteB0(
   weights='imagenet', 
   input_shape=(224,224, 3),
   include_top=False
)

model = tf.keras.Sequential([
    backbone,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)  # 10 = num classes
])
model.compile(...)
model.fit(...)
```

You can fine tune these models, just like other Keras models.  

For end-to-end fine-tuning and conversion examples check out the 
[Colab Notebook](https://colab.research.google.com/drive/1d_TJGYt68SBmCDnrNEGOz8XKbKNm_7tY?usp=sharing).

# Installation
There are multiple ways to install.  
The only requirements are Tensorflow 2.2+ and Python 3.6+.  
(Although, Tensorflow **at least** 2.4 is strongly recommended)

### Option A: (recommended) pip install from GitHub
`pip install git+https://github.com/sebastian-sz/efficientnet-lite-keras@main`

### Option B: Build from source
```bash
git clone https://github.com/sebastian-sz/efficientnet-lite-keras.git  
cd efficientnet_lite_keras  
pip install .
```

### Option C: (alternatively) no install:
If you do not want to install you could just drop the `efficientnet_lite/efficientnet_lite.py` file directly into your project.

### Option D: Docker
You can also install this package as an extension to official Tensorflow docker container:  

Build: `docker build -t efficientnet_lite_keras .`  
Run: `docker run -it --rm efficientnet_lite_keras`

For GPU support or different TAG you can (for example) pass  
`--build-arg IMAGE_TAG=2.5.0-gpu`  
in build command.

### Verify installation
If all goes well you should be able to import:  
`from efficientnet_lite import *` 

# How to use
There are 5 lite model variants you can use (B0-B4).

### Imagenet weights
The imagenet weights are automatically downloaded if you pass `weights="imagenet"` option while creating the models.

### Preprocessing
The models expect image values in range `-1:+1`. In more detail the preprocessing 
function (for pretrained models) looks as follows:  
```python
def preprocess(image):  # input image is in range 0-255.
    return (image - 127.00) / 128.00
```

##### (Alternatively) Preprocessing Layer:
Or you can use [Preprocessing Layer](https://keras.io/guides/preprocessing_layers/):
```python
from efficientnet_lite import get_preprocessing_layer

layer = get_preprocessing_layer()
inputs = layer(image)
```

### Input shapes
The following table shows input shapes for each model variant:

| Model variant | Input shape |
|:-------------:|:-----------:|
|       B0      | `224,224`   |
|       B1      | `240,240`   |
|       B2      | `260,260`   |
|       B3      | `280,280`   |
|       B4      | `300,300`   |

### Fine-tuning
For fine-tuning example, check out the [Colab Notebook](https://colab.research.google.com/drive/1d_TJGYt68SBmCDnrNEGOz8XKbKNm_7tY?usp=sharing).

### Tensorflow Lite
The models are TFLite compatible. You can convert them like any other Keras model:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("efficientnet_lite.tflite", "wb") as file:
  file.write(tflite_model)
```

### ONNX
The models are ONNX compatible. For ONNX Conversion you can use [tf2onnx](
https://github.com/onnx/tensorflow-onnx) package:
```python
!pip install tf2onnx==1.8.4

# Save the model in TF's Saved Model format:
model.save("my_saved_model/")

# Convert:
!python -m tf2onnx.convert \
  --saved-model my_saved_model/ \
  --output efficientnet_lite.onnx
```

### TF's Model Optimization Toolkit
Lite model variants were intended for mobile use and embedded systems, so I tested if
they work with Tensorflow Model Optimization Toolkit.

For example, preparing the model for pruning should work:
```python
import tensorflow_model_optimization as tfmot
from efficientnet_lite import EfficientNetLiteB0

model = EfficientNetLiteB0()
model = tfmot.sparsity.keras.prune_low_magnitude(model)
```

# Original Weights
The original weights are present in the
[original repository](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/)
for Efficient Net Lite in the form of Tensorflow's `.ckpt` files. Also, on Tensorflow's
GitHub, there is a [utility script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/efficientnet_weight_update_util.py)
for converting EfficientNet weights.

The scripts worked for me, after I modified the model's architecture, to match the
description of Lite variants.

### (Optionally) Convert the weights
The converted weights are on this repository's GitHub. If, for some reason, you wish to 
download and convert original weights yourself, I prepared the utility scripts: 
1. `bash scripts/download_all_weights.sh`
2. `bash scripts/convert_all_weights.sh`

# Bibliography
[1] [Original repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)  
[2] Existing non-lite [Keras EfficientNet models](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/efficientnet.py)  
[3] Weight update [util](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/efficientnet_weight_update_util.py)  

# Closing words
If you found this repo useful, please consider giving it a star!
