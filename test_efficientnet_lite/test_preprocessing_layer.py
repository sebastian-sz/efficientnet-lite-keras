import tensorflow as tf
from absl.testing import absltest

from efficientnet_lite import get_preprocessing_layer


class TestPreprocessingLayer(absltest.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    layer = get_preprocessing_layer()

    def test_layer_output_correct_values(self):
        mock_frame = self.rng.uniform((1, 224, 224, 3), maxval=255, dtype=tf.float32)

        expected_output = self._original_preprocessing(mock_frame)
        layer_output = self.layer(mock_frame)

        tf.debugging.assert_near(expected_output, layer_output)

    @staticmethod
    def _original_preprocessing(image):
        return (image - 127.00) / 128.00


if __name__ == "__main__":
    absltest.main()
