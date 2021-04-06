import unittest

import numpy as np
import tensorflow as tf

TOLERANCE = 1e-3

from official_eff_net import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4
)


class OutputConsistencyTestSuite:
    IMAGE_PATH = "assets/panda.jpg"

    # Override in children
    model = None
    input_shape = None
    original_model_output = None

    def test_output_consistency(self):
        img = tf.image.decode_jpeg(tf.io.read_file(self.IMAGE_PATH))
        img = tf.image.resize(img, self.input_shape)
        img = tf.expand_dims(img, axis=0)
        output = self.model.predict(img)

        expected_out = np.load(self.original_model_output)

        np.testing.assert_allclose(expected_out, output, rtol=TOLERANCE, atol=TOLERANCE)


class TestEfficientNetB0Output(unittest.TestCase, OutputConsistencyTestSuite):
    model = EfficientNetB0()
    input_shape = (224, 224)
    original_model_output = "assets/original_outputs/b0_output.npy"


class TestEfficientNetB1Output(unittest.TestCase, OutputConsistencyTestSuite):
    model = EfficientNetB1()
    input_shape = (240, 240)
    original_model_output = "assets/original_outputs/b1_output.npy"


class TestEfficientNetB2Output(unittest.TestCase, OutputConsistencyTestSuite):
    model = EfficientNetB2()
    input_shape = (260, 260)
    original_model_output = "assets/original_outputs/b2_output.npy"


class TestEfficientNetB3Output(unittest.TestCase, OutputConsistencyTestSuite):
    model = EfficientNetB3()
    input_shape = (300, 300)
    original_model_output = "assets/original_outputs/b3_output.npy"


class TestEfficientNetB4Output(unittest.TestCase, OutputConsistencyTestSuite):
    model = EfficientNetB4()
    input_shape = (380, 380)
    original_model_output = "assets/original_outputs/b4_output.npy"


if __name__ == '__main__':
    unittest.main()
