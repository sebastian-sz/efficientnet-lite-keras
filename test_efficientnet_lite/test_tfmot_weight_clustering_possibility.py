from typing import Callable, Tuple

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized

from test_efficientnet_lite.test_model import TEST_PARAMS

# Disable GPU
tf.config.set_visible_devices([], "GPU")


class TestWeightClusteringWrappers(parameterized.TestCase):
    centroid_initialization = tfmot.clustering.keras.CentroidInitialization
    clustering_params = {
        "number_of_clusters": 3,
        "cluster_centroids_init": centroid_initialization.DENSITY_BASED,
    }

    def setUp(self):
        tf.keras.backend.clear_session()

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tfmot_weight_clustering_wrap(
        self, model_fn: Callable, input_shape: Tuple[int, int]
    ):
        model = model_fn(weights=None, input_shape=input_shape + (3,))
        tfmot.clustering.keras.cluster_weights(model, **self.clustering_params)


if __name__ == "__main__":
    absltest.main()
