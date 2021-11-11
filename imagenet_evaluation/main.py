import tensorflow as tf
from absl import app, flags
from external.imagenet_input import ImageNetInput

from efficientnet_lite import (
    EfficientNetLiteB0,
    EfficientNetLiteB1,
    EfficientNetLiteB2,
    EfficientNetLiteB3,
    EfficientNetLiteB4,
    get_preprocessing_layer,
)

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "variant",
    default="b0",
    enum_values=["b0", "b1", "b2", "b3", "b4"],
    help="Which model variant to evaluate.",
)
flags.DEFINE_string(
    "data_dir",
    default="/workspace/tfrecords/validation",
    help="Path to validation tfrecords.",
)
flags.DEFINE_integer("batch_size", default=16, help="Batch size for eval.")
flags.DEFINE_bool(
    "use_tfhub",
    default=False,
    help="Whether to evaluate TFHub models instead of this repo.",
)

VARIANT_TO_MODEL = {
    "b0": EfficientNetLiteB0,
    "b1": EfficientNetLiteB1,
    "b2": EfficientNetLiteB2,
    "b3": EfficientNetLiteB3,
    "b4": EfficientNetLiteB4,
}

VARIANT_TO_HUB_URL = {
    "b0": "https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2",
    "b1": "https://tfhub.dev/tensorflow/efficientnet/lite1/classification/2",
    "b2": "https://tfhub.dev/tensorflow/efficientnet/lite2/classification/2",
    "b3": "https://tfhub.dev/tensorflow/efficientnet/lite3/classification/2",
    "b4": "https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2",
}

VARIANT_TO_INPUT_SHAPE = {"b0": 224, "b1": 240, "b2": 260, "b3": 280, "b4": 300}


def main(argv_):
    """Run Imagenet Eval job."""
    # Load model
    image_size = VARIANT_TO_INPUT_SHAPE[FLAGS.variant]
    if FLAGS.use_tfhub:
        import tensorflow_hub as tf_hub  # Local import so this is entirely optional

        hub_url = VARIANT_TO_HUB_URL[FLAGS.variant]
        model = tf_hub.KerasLayer(hub_url)
        model.build([None, image_size, image_size, 3])
    else:
        model = VARIANT_TO_MODEL[FLAGS.variant](input_shape=(image_size, image_size, 3))

    # Load data
    params = {"batch_size": FLAGS.batch_size}
    # https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/efficientnet/main.py#L648
    resize_method = tf.image.ResizeMethod.BILINEAR

    val_dataset = ImageNetInput(
        data_dir=FLAGS.data_dir,
        is_training=False,
        image_size=image_size,
        transpose_input=False,
        use_bfloat16=False,
        resize_method=resize_method,
    ).input_fn(params=params)

    if FLAGS.use_tfhub:
        val_dataset = val_dataset.map(lambda img, label: (img / 255.0, label))
    else:
        preprocessing_layer = get_preprocessing_layer()
        val_dataset = val_dataset.map(
            lambda img, label: (preprocessing_layer(img), label)
        )

    # Run eval:
    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top1")
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")
    progbar = tf.keras.utils.Progbar(target=50000 // FLAGS.batch_size)

    for idx, (images, y_true) in enumerate(val_dataset):
        y_pred = model(images, training=False)

        top1.update_state(y_true=y_true, y_pred=y_pred)
        top5.update_state(y_true=y_true, y_pred=y_pred)

        progbar.update(
            idx, [("top1", top1.result().numpy()), ("top5", top5.result().numpy())]
        )

    print()
    print(f"TOP1: {top1.result().numpy()}.  TOP5: {top5.result().numpy()}")


if __name__ == "__main__":
    app.run(main)
