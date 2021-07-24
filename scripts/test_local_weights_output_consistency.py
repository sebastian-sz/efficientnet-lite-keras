"""Test script to check if locally converted weights are OK."""
import os
from collections import Callable
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
from tests._root_dir import ROOT_DIR

WEIGHTS_DIR = "/".join(ROOT_DIR.split("/")[:-1]) + "/weights"

LOCAL_OUTPUT_CONSISTENCY_TEST_PARAMS = [
    {
        "testcase_name": "b0",
        "model_fn": EfficientNetLiteB0,
        "input_shape": (224, 224),
        "weights_path": os.path.join(WEIGHTS_DIR, "efficientnet_lite_b0.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b0_lite_output_224.npy"
        ),
    },
    {
        "testcase_name": "b1",
        "model_fn": EfficientNetLiteB1,
        "input_shape": (240, 240),
        "weights_path": os.path.join(WEIGHTS_DIR, "efficientnet_lite_b1.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b1_lite_output_240.npy"
        ),
    },
    {
        "testcase_name": "b2",
        "model_fn": EfficientNetLiteB2,
        "input_shape": (260, 260),
        "weights_path": os.path.join(WEIGHTS_DIR, "efficientnet_lite_b2.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b2_lite_output_260.npy"
        ),
    },
    {
        "testcase_name": "b3",
        "model_fn": EfficientNetLiteB3,
        "input_shape": (280, 280),
        "weights_path": os.path.join(WEIGHTS_DIR, "efficientnet_lite_b3.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b3_lite_output_280.npy"
        ),
    },
    {
        "testcase_name": "b4",
        "model_fn": EfficientNetLiteB4,
        "input_shape": (300, 300),
        "weights_path": os.path.join(WEIGHTS_DIR, "efficientnet_lite_b4.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b4_lite_output_300.npy"
        ),
    },
]

LOCAL_FEATURE_EXTRACTION_TEST_PARAMS = [
    {
        "testcase_name": "b0-fe",
        "model_fn": EfficientNetLiteB0,
        "input_shape": (224, 224),
        "weights_path": os.path.join(WEIGHTS_DIR, "efficientnet_lite_b0_notop.h5"),
    },
    {
        "testcase_name": "b1-fe",
        "model_fn": EfficientNetLiteB1,
        "input_shape": (240, 240),
        "weights_path": os.path.join(WEIGHTS_DIR, "efficientnet_lite_b1_notop.h5"),
    },
    {
        "testcase_name": "b2-fe",
        "model_fn": EfficientNetLiteB2,
        "input_shape": (260, 260),
        "weights_path": os.path.join(WEIGHTS_DIR, "efficientnet_lite_b2_notop.h5"),
    },
    {
        "testcase_name": "b3-fe",
        "model_fn": EfficientNetLiteB3,
        "input_shape": (280, 280),
        "weights_path": os.path.join(WEIGHTS_DIR, "efficientnet_lite_b3_notop.h5"),
    },
    {
        "testcase_name": "b4-fe",
        "model_fn": EfficientNetLiteB4,
        "input_shape": (300, 300),
        "weights_path": os.path.join(WEIGHTS_DIR, "efficientnet_lite_b4_notop.h5"),
    },
]


class TestKerasVSOriginalOutputConsistency(parameterized.TestCase):
    image_path = os.path.join(ROOT_DIR, "assets/panda.jpg")
    image = tf.image.decode_png(tf.io.read_file(image_path))
    image = tf.expand_dims(image, axis=0)

    @parameterized.named_parameters(LOCAL_OUTPUT_CONSISTENCY_TEST_PARAMS)
    def test_keras_and_original_outputs_the_same(
        self,
        model_fn: Callable,
        input_shape: Tuple[int, int],
        weights_path: str,
        original_outputs: str,
    ):
        if not os.path.exists(weights_path):
            self.skipTest("No weights present in repo. Skipping... .")

        model = model_fn(weights=None)
        model.load_weights(weights_path)

        input_tensor = tf.image.resize(self.image, input_shape)
        input_tensor = self._pre_process_image(input_tensor)

        output = model(input_tensor, training=False)
        original_output = np.load(original_outputs)

        tf.debugging.assert_near(output, original_output)

    @staticmethod
    def _pre_process_image(img: tf.Tensor) -> tf.Tensor:
        return (img - 127.00) / 128.00

    @parameterized.named_parameters(LOCAL_FEATURE_EXTRACTION_TEST_PARAMS)
    def test_local_notop_weights_for_feature_extraction(
        self, model_fn: Callable, input_shape: Tuple[int, int], weights_path: str
    ):
        if not os.path.exists(weights_path):
            self.skipTest("No weights present in repo. Skipping... .")

        model = model_fn(weights=None, include_top=False)
        model.load_weights(weights_path)

        input_tensor = tf.image.resize(self.image, input_shape)
        input_tensor = self._pre_process_image(input_tensor)

        output = model(input_tensor, training=False)

        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertEqual(len(output.shape), 4)


if __name__ == "__main__":
    absltest.main()
