import os
import tempfile
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from test_efficientnet_lite import utils
from test_efficientnet_lite.test_model import TEST_PARAMS

# Disable GPU
tf.config.set_visible_devices([], "GPU")


class TestTFLiteConversion(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    tflite_path = os.path.join(tempfile.mkdtemp(), "model.tflite")

    def tearDown(self) -> None:
        if os.path.exists(self.tflite_path):
            os.remove(self.tflite_path)

    def setUp(self):
        tf.keras.backend.clear_session()

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tflite_conversion(self, model_fn: Callable, input_shape: Tuple[int, int]):
        model = model_fn(weights=None, input_shape=(*input_shape, 3))

        self._convert_and_save_tflite(model, input_shape)

        # Verify outputs:
        dummy_inputs = self.rng.uniform(shape=(1, *input_shape, 3))
        tflite_output = self._run_tflite_inference(dummy_inputs)
        self.assertTrue(isinstance(tflite_output, np.ndarray))
        self.assertEqual(tflite_output.shape, (1, 1000))

    def _convert_and_save_tflite(
        self, model: tf.keras.Model, input_shape: Tuple[int, int]
    ):
        inference_func = utils.get_inference_function(model, input_shape)
        concrete_func = inference_func.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

        tflite_model = converter.convert()
        with open(self.tflite_path, "wb") as file:
            file.write(tflite_model)

    def _run_tflite_inference(self, inputs: tf.Tensor) -> np.ndarray:
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], inputs.numpy())
        interpreter.invoke()

        return interpreter.get_tensor(output_details[0]["index"])


if __name__ == "__main__":
    absltest.main()
