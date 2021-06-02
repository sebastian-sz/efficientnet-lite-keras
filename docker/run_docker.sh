docker run \
  -it \
  --rm \
  -w /workspace/ \
  -u "$(id -u)":"$(id -g)" \
  -v $PWD:/workspace \
  keras_efficientnet_lite