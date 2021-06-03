import os
import tempfile
from typing import Tuple

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from tests.test_efficientnet_lite import TEST_PARAMS

# Disable GPU
tf.config.set_visible_devices([], "GPU")


class TestTFLiteConversion(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    tflite_path = os.path.join(tempfile.mkdtemp(), "model.tflite")

    def tearDown(self) -> None:
        if os.path.exists(self.tflite_path):
            os.remove(self.tflite_path)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tflite_conversion(
        self, model: tf.keras.Model, input_shape: Tuple[int, int]
    ):
        model = model(weights=None, input_shape=(*input_shape, 3))

        self._convert_and_save_tflite(model)
        self.assertTrue(os.path.isfile(self.tflite_path))

        # Check outputs:
        mock_input = self.rng.uniform(shape=(1, *input_shape, 3), dtype=tf.float32)
        original_output = model.predict(mock_input)
        tflite_output = self._run_tflite_inference(mock_input)

        tf.debugging.assert_near(original_output, tflite_output, rtol=1e-3, atol=1e-3)

    def _convert_and_save_tflite(self, model: tf.keras.Model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
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
