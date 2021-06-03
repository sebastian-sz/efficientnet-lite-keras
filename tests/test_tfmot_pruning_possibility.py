import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized

from tests.test_efficientnet_lite import TEST_PARAMS

# Disable GPU
tf.config.set_visible_devices([], "GPU")


class TestEfficientNetLitePruningWrapper(parameterized.TestCase):
    @parameterized.named_parameters(TEST_PARAMS)
    def test_tfmot_pruning_wrapper(self, model: tf.keras.Model, **kwargs):
        model = model(weights=None)
        tfmot.sparsity.keras.prune_low_magnitude(model)


if __name__ == "__main__":
    absltest.main()
