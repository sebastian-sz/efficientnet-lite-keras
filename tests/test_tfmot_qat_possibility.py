import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized

from tests.test_efficientnet_lite import TEST_PARAMS

# Disable GPU
tf.config.set_visible_devices([], "GPU")


class TestEfficientNetLiteQATWrap(parameterized.TestCase):
    @parameterized.named_parameters(TEST_PARAMS)
    def test_qat_wrapper(self, model: tf.keras.Model, **kwargs):
        model = model(weights=None)
        tfmot.quantization.keras.quantize_model(model)


if __name__ == "__main__":
    absltest.main()
