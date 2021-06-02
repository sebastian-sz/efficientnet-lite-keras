import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from efficientnet_lite import (
    EfficientNetLiteB0,
    EfficientNetLiteB1,
    EfficientNetLiteB2,
    EfficientNetLiteB3,
    EfficientNetLiteB4,
)
from efficientnet_lite.tests._root_dir import ROOT_DIR

# Disable GPU
tf.config.set_visible_devices([], "GPU")

OUTPUT_TEST_PARAMS = [
    {
        "testcase_name": "b0",
        "model": EfficientNetLiteB0,
        "input_shape": (224, 224),
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b0_lite_output_224.npy"
        ),
    },
    {
        "testcase_name": "b1",
        "model": EfficientNetLiteB1,
        "input_shape": (240, 240),
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b1_lite_output_240.npy"
        ),
    },
    {
        "testcase_name": "b2",
        "model": EfficientNetLiteB2,
        "input_shape": (260, 260),
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b2_lite_output_260.npy"
        ),
    },
    {
        "testcase_name": "b3",
        "model": EfficientNetLiteB3,
        "input_shape": (280, 280),
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b3_lite_output_280.npy"
        ),
    },
    {
        "testcase_name": "b4",
        "model": EfficientNetLiteB4,
        "input_shape": (300, 300),
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b4_lite_output_300.npy"
        ),
    },
]


class TestKerasVSOriginalOutputConsistency(parameterized.TestCase):
    image_path = os.path.join(ROOT_DIR, "assets/panda.jpg")
    image = tf.image.decode_png(tf.io.read_file(image_path))
    image = tf.expand_dims(image, axis=0)

    @parameterized.named_parameters(OUTPUT_TEST_PARAMS)
    def test_keras_and_original_outputs_the_same(
        self, model: tf.keras.Model, input_shape: Tuple[int, int], original_outputs: str
    ):

        model = model(weights="imagenet", input_shape=input_shape + (3,))
        inputs = tf.image.resize(self.image, size=input_shape)
        inputs = self._pre_process_image(inputs)
        outputs = model.predict(inputs)

        original_outputs = np.load(original_outputs)

        tf.debugging.assert_near(outputs, original_outputs)

    @staticmethod
    def _pre_process_image(img: tf.Tensor) -> tf.Tensor:
        return (img - 127.00) / 128.00


if __name__ == "__main__":
    absltest.main()
