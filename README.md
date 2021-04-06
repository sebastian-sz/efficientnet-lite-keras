# efficientnet-lite-keras

Todos:

0) Setup dev Dockerfile.
1) Add conversion script from tf.keras.applications
2) Add ugly conversion script from keras-applications github
3) Maintain the weights in Google Drive.
4) Add tests for comparing original repo / this repo output
5) Add scripts to reproduce outputs.
6) Add tests regarding conversion.
7) Add how to use load those models.
8) Figure out how to change Documentation.
	a) Input shapes differ.
	b) The input range values is the same (0-255) = preprocessing is different but is still a part of the model.
	c) Lite flag only in models from B0-B4. Raise Value Error in heavier models.

9) Test whether one can still call old models without the lite flag.
