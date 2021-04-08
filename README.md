# Introduction
This is a rather utility repo for converiting (and loading) EfficientNet Lite weights
from Tensorflow's `.ckpt` format into Keras's `.h5` files.   

Those weights are compatible and can be used similar to other models present in 
`tf.keras.applications`.

# How to convert weights:
The steps are as follows:

1. Start Tensorflow docker: `bash run_docker.sh`
2. Download original TF's ckpt weights: `cd weights/original_weights` and 
   `bash download_all.sh`. This will download weights for B0Lite - B4Lite variants.
4. `cd ../..`
5. You can convert weights via modified script `efficientnet_weight_update_util.py`. For
	detailed usage please refer to possible args in the script. To convert all variants
   (B0 - B4) you can use the `bash convert_all_weights.sh` utility script.
6. In `weights/` you should have all the converted "lite" weights in Keras `.h5` format.
7. Run tests if converted weights produce the same output as original model variants
   `python test_output_consistency.py` (this takes ~15 seconds).

# How to use EfficientNetLite?

You can call the existing models via:
```python
from efficientnet import EfficientNetB0

model = EfficientNetB0(lite=True, weights="path/to/converted/h5_file")
```
The model expects RGB Frame in range 0-255 (just like the non-lite variant).
The input shapes are:
```
B0Lite: 224x224
B1Lite: 240x240
B2Lite: 260x260 
B3Lite: 280x280  # NOT 300 as opposed to non-lite.
B4Lite: 300x300  # NOT 380 as opposed to non-lite.
```

# Generating original outputs:
For sanity checking I ran inference on panda image (also mentioned in original 
EfficientNet repository) and saved the results as numpy (`.npy`) files.

The inference was run using the lite variants - to check if weights and 
architecture were successfully ported.

The inference was also run using the non-lite variants - for sanity checking if the 
original models remained unchanged.

To obtain the original, non-lite model outputs: you can run   
`cd utils; python generate_original_outputs.py`

To obtain the original, lite model outputs I provided a Colab Notebook that is sadly
rather manual. Inspecting `utils/efficient_net_lite_original_outputs.ipynb` should
give you a general idea.


# Further todos:

3) (optional) Maintain the weights in Google Drive in case authors decide to get rid of them.
8) Lite flag only in models from B0-B4. Raise Value Error in heavier models as `lite` 
   option can be (but will throw errors) passed via `kwargs`?
