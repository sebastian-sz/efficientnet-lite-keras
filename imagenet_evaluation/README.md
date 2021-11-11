# Imagenet Evaluation
This directory contains code and results for Imagenet Evaluation as a final 
sanity check regarding the rewrite + weight conversion.

### Results

#### Notice
Please, bear in mind that:

* the models in this repo produce the same logits/probabilities
as the ones exported from the original repository, using the official checkpoint files.
* [Tensorflow Hub models](https://tfhub.dev/s?q=EfficientNet-Lite),
produce **exactly the same** (up to 13th decimal) Imagenet accuracy results.
* There are [issues](https://github.com/tensorflow/tpu/issues/882) by other people who also have trouble reproducing Imagenet validation accuracy.


#### Results table
| Variant | Image size | Reported Top 1 | This repo Top 1 | This repo Top 5 | Top 1 difference |
| ------- | ---------- | -------------- | --------------- | --------------- | ---------------- |
|   B0    |     224    |     75.1       |       75.1      |       92.3      |       0.0        |
|   B1    |     240    |     76.7       |       76.8      |       93.3      |       +0.1       |
|   B2    |     260    |     77.6       |       77.6      |       93.8      |       0.0        |
|   B3    |     280    |     79.8       |       79.4*     |       94.7      |       -0.4*      |
|   B4    |     300    |     81.5       |       80.8*     |       95.2      |       -0.7*      |

*Mismatched Accuracy for B3 and B4 variant. See [Notice](https://github.com/sebastian-sz/efficientnet-lite-keras/tree/main/imagenet_evaluation#notice) above.


#### Why small differences?
One can speculate. The differences might come from:
* The API used: Official uses `TPUEstimator`, I use `tf.keras.Model`
* Hardware used: Official uses TPU, I use GPU.
* Precision: Official runs in `bfloat16`, I use `float32`.

### To reproduce my eval:
1. Download Imagenet and use [imagenet_to_gcs](https://github.com/tensorflow/tpu/blob/acb331c8878ce5a4124d4d7687df5fe0fadcd43b/tools/datasets/imagenet_to_gcs.py) script to obtain tfrecords.
```
python imagenet_to_gcs.py \
    --raw_data_dir=imagenet_home/ \
    --local_scratch_dir=my_tfrecords \
    --nogcs_upload
```
        

2. To eval this repo models, run   
```python
python main.py \
    --variant b0 \
    --data_dir /path/to/tfrecords/validation
```
Change `--variant` accordingly.

3. If you want to evaluate Tensorflow hub models, add `--use_tfhub` flag.
