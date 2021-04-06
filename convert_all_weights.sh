python efficientnet_weight_update_util.py \
  --model b0 \
  --ckpt weights/original_weights/efficientnet-lite0/model.ckpt \
  --lite \
  --o weights/efficient_net_lite_b0.h5

python efficientnet_weight_update_util.py \
  --model b1 \
  --ckpt weights/original_weights/efficientnet-lite1/model.ckpt \
  --lite \
  --o weights/efficient_net_lite_b1.h5

python efficientnet_weight_update_util.py \
  --model b2 \
  --ckpt weights/original_weights/efficientnet-lite2/model.ckpt \
  --lite \
  --o weights/efficient_net_lite_b2.h5

python efficientnet_weight_update_util.py \
  --model b3 \
  --ckpt weights/original_weights/efficientnet-lite3/model.ckpt \
  --lite \
  --o weights/efficient_net_lite_b3.h5

python efficientnet_weight_update_util.py \
  --model b4 \
  --ckpt weights/original_weights/efficientnet-lite4/model.ckpt \
  --lite \
  --o weights/efficient_net_lite_b4.h5