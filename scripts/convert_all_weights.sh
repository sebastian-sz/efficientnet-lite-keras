python scripts/efficientnet_weight_update_util.py \
  --model b0 \
  --ckpt weights/original_weights/efficientnet-lite0/model.ckpt \
  -o weights/efficientnet_lite_b0.h5

python scripts/efficientnet_weight_update_util.py \
  --model b1 \
  --ckpt weights/original_weights/efficientnet-lite1/model.ckpt \
  -o weights/efficientnet_lite_b1.h5

python scripts/efficientnet_weight_update_util.py \
  --model b2 \
  --ckpt weights/original_weights/efficientnet-lite2/model.ckpt \
  -o weights/efficientnet_lite_b2.h5

python scripts/efficientnet_weight_update_util.py \
  --model b3 \
  --ckpt weights/original_weights/efficientnet-lite3/model.ckpt \
  -o weights/efficientnet_lite_b3.h5

python scripts/efficientnet_weight_update_util.py \
  --model b4 \
  --ckpt weights/original_weights/efficientnet-lite4/model.ckpt \
  -o weights/efficientnet_lite_b4.h5

python scripts/efficientnet_weight_update_util.py \
  --model b0 \
  --ckpt weights/original_weights/efficientnet-lite0/model.ckpt \
  --notop \
  -o weights/efficientnet_lite_b0_notop.h5

python scripts/efficientnet_weight_update_util.py \
  --model b1 \
  --ckpt weights/original_weights/efficientnet-lite1/model.ckpt \
  --notop \
  -o weights/efficientnet_lite_b1_notop.h5

python scripts/efficientnet_weight_update_util.py \
  --model b2 \
  --ckpt weights/original_weights/efficientnet-lite2/model.ckpt \
  --notop \
  -o weights/efficientnet_lite_b2_notop.h5

python scripts/efficientnet_weight_update_util.py \
  --model b3 \
  --ckpt weights/original_weights/efficientnet-lite3/model.ckpt \
  --notop \
  -o weights/efficientnet_lite_b3_notop.h5

python scripts/efficientnet_weight_update_util.py \
  --model b4 \
  --ckpt weights/original_weights/efficientnet-lite4/model.ckpt \
  --notop \
  -o weights/efficientnet_lite_b4_notop.h5
