docker run \
  -it \
  --rm \
  -w /workspace/ \
  -v $PWD:/workspace \
  -u $(id -u):$(id -g) \
  tensorflow/tensorflow:2.4.1