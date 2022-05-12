import os
import tempfile
from typing import Callable, Tuple

import numpy as np
import onnxruntime
import tensorflow as tf
import tf2onnx
from absl.testing import absltest, parameterized

from test_efficientnet_lite import utils
from test_efficientnet_lite.test_model import TEST_PARAMS

# Disable GPU
tf.config.set_visible_devices([], "GPU")


class TestONNXConversion(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    onnx_model_path = os.path.join(tempfile.mkdtemp(), "model.onnx")

    def tearDown(self) -> None:
        if os.path.exists(self.onnx_model_path):
            os.remove(self.onnx_model_path)

    def setUp(self):
        tf.keras.backend.clear_session()

    @parameterized.named_parameters(TEST_PARAMS)
    def test_model_onnx_conversion(
        self, model_fn: Callable, input_shape: Tuple[int, int]
    ):
        model = model_fn(weights=None, input_shape=(*input_shape, 3))

        inference_func = utils.get_inference_function(model, input_shape)
        self._convert_onnx(inference_func)

        # Verify output:
        dummy_inputs = self.rng.uniform(shape=(1, *input_shape, 3), dtype=tf.float32)
        onnx_session = onnxruntime.InferenceSession(self.onnx_model_path)
        onnx_inputs = {onnx_session.get_inputs()[0].name: dummy_inputs.numpy()}
        onnx_output = onnx_session.run(None, onnx_inputs)[0]

        self.assertTrue(isinstance(onnx_output, np.ndarray))
        self.assertEqual(onnx_output.shape, (1, 1000))

    def _convert_onnx(self, inference_func):
        model_proto, _ = tf2onnx.convert.from_function(
            inference_func,
            output_path=self.onnx_model_path,
            input_signature=inference_func.input_signature,
        )
        return model_proto


if __name__ == "__main__":
    absltest.main()
