# Table of contens
1. [Introduction](https://github.com/sebastian-sz/efficientnet-lite-keras#introduction)
2. [Quickstart](https://github.com/sebastian-sz/efficientnet-lite-keras#quickstart)
3. [Installation](https://github.com/sebastian-sz/efficientnet-lite-keras#installation)
4. [Usage](https://github.com/sebastian-sz/efficientnet-lite-keras#usage)
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

# Use model directly:
model = EfficientNetLiteB0(weights='imagenet', input_shape=(224, 224, 3))
model.summary()

# Or as feature extractor:
feature_extractor = EfficientNetLiteB0(
   weights='imagenet', 
   input_shape=(224,224, 3),
   include_top=False
)
```

You can also fine tune these models, just like other Keras models. For end-to-end fine tuning example check out the [Colab Notebook](https://colab.research.google.com/drive/1d_TJGYt68SBmCDnrNEGOz8XKbKNm_7tY?usp=sharing).

# Installation
There are multiple ways to install.  
The only requirements are Tensorflow 2.x and Python 3.6+.

### (Recommended) pip install from github
`pip install git+https://github.com/sebastian-sz/efficientnet-lite-keras@main`

### Build from source
```bash
git clone https://github.com/sebastian-sz/efficientnet-lite-keras.git  
cd efficientnet_lite_keras  
pip install .
```

### (Alternatively) No install:
If you do not want to install you could just drop the `efficientnet_lite/efficientnet_lite.py` file directly into your project.

### Verify installation
If all goes well you should be able to import:  
`from efficientnet_lite import *` 

# Usage
There are 5 lite model variants you can use (B0-B4).

### Imagenet weights
The imagenet weights are automatically downloaded if you pass `weights="imagenet"` option while creating the models.

### Preprocessing
The models expect image values in range `-1:+1`. In more detail the preprocessing function looks as follows:  
```python
def preprocess(image):  # input image is in range 0-255.
    return (image - 127.00) / 128.00
```

### Input shapes
The following table shows input shapes for each model variant:

| Model variant | Input shape |
|:-------------:|:-----------:|
|       B0      | `224,224`  |
|       B1      | `240,240`  |
|       B2      | `260,260`   |
|       B3      | `280,280`   |
|       B4      | `300,300`   |

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
[original repoistory](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/)
for Efficient Net in the form of Tensorflow's `.ckpt` files. Also, on Tensorflow's
Github, there is a [utility script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/efficientnet_weight_update_util.py)
for converting EfficientNet weights.

The scripts worked for me, after I modified the model's architecture, to match the
description of Lite variants.

### (Optionally) Convert the weights
I am hosting the converted weights on DropBox. If, for some reason, you wish to download and convert original weights yourself, I prepered the utility scripts:  

# Bibliography
[1] [Original repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)  
[2] Existing non-lite [Keras EfficientNet models](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/efficientnet.py)  
[3] Weight update [util](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/efficientnet_weight_update_util.py)  

# Closing words
If you found this repo useful, please consider giving it a star!
