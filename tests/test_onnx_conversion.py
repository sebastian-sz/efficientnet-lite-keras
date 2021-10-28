import os
import tempfile
from typing import Callable, Tuple

import onnxruntime
import tensorflow as tf
import tf2onnx
from absl.testing import absltest, parameterized
from tensorflow.python.types.core import GenericFunction

from tests.test_efficientnet_lite import TEST_PARAMS
from tests.utils import get_inference_function

# Disable GPU
tf.config.set_visible_devices([], "GPU")


class TestONNXConversion(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    onnx_model_path = os.path.join(tempfile.mkdtemp(), "model.onnx")

    _tolerance = 1e-5

    def tearDown(self) -> None:
        if os.path.exists(self.onnx_model_path):
            os.remove(self.onnx_model_path)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_model_onnx_conversion(
        self, model_fn: Callable, input_shape: Tuple[int, int]
    ):
        tf.keras.backend.clear_session()

        # Comparison will fail with random weights as we are comparing
        # very low floats:
        model = model_fn(weights="imagenet", input_shape=(*input_shape, 3))
        inference_func = get_inference_function(model, input_shape)

        self._convert_onnx(inference_func)

        self.assertTrue(os.path.isfile(self.onnx_model_path))

        # Compare outputs:
        mock_input = self.rng.uniform(shape=(1, *input_shape, 3), dtype=tf.float32)
        original_output = model(mock_input, training=False)

        onnx_session = onnxruntime.InferenceSession(self.onnx_model_path)
        onnx_inputs = {onnx_session.get_inputs()[0].name: mock_input.numpy()}
        onnx_output = onnx_session.run(None, onnx_inputs)

        tf.debugging.assert_near(
            original_output, onnx_output, rtol=self._tolerance, atol=self._tolerance
        )

    def _convert_onnx(self, inference_func: GenericFunction):
        model_proto, _ = tf2onnx.convert.from_function(
            inference_func,
            output_path=self.onnx_model_path,
            input_signature=inference_func.input_signature,
        )
        return model_proto


if __name__ == "__main__":
    absltest.main()
