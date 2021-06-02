import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import parameterized

from efficientnet_lite.tests.test_efficientnet_lite import TEST_PARAMS

# Disable GPU
tf.config.set_visible_devices([], "GPU")


class TestWeightClusteringWrappers(parameterized.TestCase):
    centroid_initialization = tfmot.clustering.keras.CentroidInitialization
    clustering_params = {
        "number_of_clusters": 3,
        "cluster_centroids_init": centroid_initialization.DENSITY_BASED,
    }

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tfmot_weight_clustering_wrap(self, model: tf.keras.Model, **kwargs):
        model = model(weights=None)
        tfmot.clustering.keras.cluster_weights(model, **self.clustering_params)
