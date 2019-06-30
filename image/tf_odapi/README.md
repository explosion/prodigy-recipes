# Custom Recipes using Tensorflow's Object Detection API

## Contents

* [docs](./docs): A folder containing documentaions as `markdown` files
* [misc](./misc): A folder containing few miscellaneous scripts.
* [image_frozen_model.py](./image_frozen_model.py): Contains the `image.frozenmodel` recipe which is basically a model in loop annotation recipe using an object detection model. This uses the `frozen graph`, either from [Tensorflow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) or from your custom export.
* [image_tf_serving.py](./image_tf_serving.py): Contains the `image.servingmodel` recipe. This is similar to above but, uses `SavedModel` instead of the `frozen graph`. As the name suggests, this recipe requires [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving).
* [image_train.py](./image_train.py): Contains the `image.trainmodel` recipe. This supports taining the Object Detection API models in loop. See [training_model_in_loop.md](./docs/training_model_in_loop.md) for documentation.
