# Introduction
This is a package with EfficientNet-Lite model variants adapted to Keras.

# How to use
There are 5 lite model variants you can use: B0, B1, B2, B3, B4.

### Installation
TODO

### Usage


The functionality is basically the same as for other models in
`tf.keras.applications`. The snippet:
```python
from efficientnet_lite import EfficientNetLiteB0

model = EfficientNetLiteB0(weights='imagenet', input_shape=(224, 224, 3))
```

will create the B0 model variant and download Imagenet weights.

Fine-tuning tutorial coming soon! (TODO)

### Preprocessing
The models expect image values in range `-1:+1`. In more detail the preprocessing
function looks as follows:
```python
def preprocess(image):
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

Pruning tutorial coming soon! (TODO)

# Original weights
The original weights are present in the
[original repoistory](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/)
for Efficient Net in the form of Tensorflow's `.ckpt` files. Also, on Tensorflow's
Github, there is a [utility script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/efficientnet_weight_update_util.py)
for converting EfficientNet weights.

The scripts worked for me, after I modified the model's architecture, to match the
description of Lite variants.

### Convert the weights manually:
I am hosting the converted weights on DropBox. They will be automatically downloaded
if you pass `weights="imagenet"` argument when creating the model.

If you want to convert the weights
yourself, I provided utility scripts to do so:

   1. `bash scripts/download_all_weights.sh` will download the original ckpt files into
`weights/original_weights` directory.
   2. `bash/scripts/convert_all_weights.sh` will convert all downloaded weights into
      Keras's `.h5` files.

### The architecture
The differences between Lite and non-Lite variants are as follows:
* Remove squeeze-and-excite.
* Replace all swish with RELU6.
* Fix the stem and head while scaling models up.

# Bibliography
[1] [Original repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)
[2] Existing non-lite [Keras EfficientNet models](TODO)
[3] Weight update [util](TODO)

# Closing words
If you found this repo useful, please consider giving it a star!
