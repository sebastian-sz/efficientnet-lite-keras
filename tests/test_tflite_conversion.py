import os
import tempfile
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from tests.test_efficientnet_lite import TEST_PARAMS
from tests.utils import get_inference_function

# Disable GPU
tf.config.set_visible_devices([], "GPU")


class TestTFLiteConversion(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    tflite_path = os.path.join(tempfile.mkdtemp(), "model.tflite")

    _tolerance = 1e-5

    def tearDown(self) -> None:
        if os.path.exists(self.tflite_path):
            os.remove(self.tflite_path)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tflite_conversion(self, model_fn: Callable, input_shape: Tuple[int, int]):
        tf.keras.backend.clear_session()

        # Comparison will fail with random weights as we are comparing
        # very low floats:
        model = model_fn(weights="imagenet", input_shape=(*input_shape, 3))

        self._convert_and_save_tflite(model, input_shape)
        self.assertTrue(os.path.isfile(self.tflite_path))

        # Check outputs:
        mock_input = self.rng.uniform(shape=(1, *input_shape, 3))

        original_output = model(mock_input, training=False)
        tflite_output = self._run_tflite_inference(mock_input)

        tf.debugging.assert_near(
            original_output, tflite_output, rtol=self._tolerance, atol=self._tolerance
        )

    def _convert_and_save_tflite(
        self, model: tf.keras.Model, input_shape: Tuple[int, int]
    ):
        inference_func = get_inference_function(model, input_shape)
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
