from typing import Tuple

import numpy as np
import tensorflow as tf

IMG_PATH = "../assets/panda.jpg"  # This will crash if ran from a different directory.


def generate_output(
        model: tf.keras.Model, input_shape: Tuple[int, int], output_path: str
):
    model = model(weights='imagenet')

    img = tf.image.decode_jpeg(tf.io.read_file(IMG_PATH))
    img = tf.image.resize(img, input_shape)
    img = tf.expand_dims(img, axis=0)
    output = model.predict(img)

    np.save(output_path, output)


def main():
    models = [
        {
            "model": tf.keras.applications.EfficientNetB0,
            "input_shape": (224, 224),
            "output_dir": "../assets/original_outputs/b0_output.npy",
        },
        {
            "model": tf.keras.applications.EfficientNetB1,
            "input_shape": (240, 240),
            "output_dir": "../assets/original_outputs/b1_output.npy",
        },
        {
            "model": tf.keras.applications.EfficientNetB2,
            "input_shape": (260, 260),
            "output_dir": "../assets/original_outputs/b2_output.npy",
        },
        {
            "model": tf.keras.applications.EfficientNetB3,
            "input_shape": (300, 300),
            "output_dir": "../assets/original_outputs/b3_output.npy",
        },
        {
            "model": tf.keras.applications.EfficientNetB4,
            "input_shape": (380, 380),
            "output_dir": "../assets/original_outputs/b4_output.npy",
        }
    ]

    for model in models:
        generate_output(model["model"], model["input_shape"], model["output_dir"])


if __name__ == '__main__':
    main()
