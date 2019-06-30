# Training the Tensorflow Object Detection API model in loop

This document explains how to use [image.trainmodel](../image_train.py) recipe to train object detection models from Tensorflow Object Detection API with Prodigy. To run this recipe you will need [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving).

`prodigy image.trainmodel -F $PATH_TO/image_train.py --help` to see the **arguments** for the recipe

Running this recipe will create the following 3 folders if not already present:
*   An **export** directory where the models used by Tensorflow Serving will be saved. Specified by `export_dir` argument.
*   A **model** directory where trained model checkpoints and Tensorboard events are stored. Specified by `model_dir` argument.
*   A **data** directory where the **TF-Records** for training are stored. Specified by `data_dir` argument.

## Recipe Flow:
The general flow of the recipe is as follows:

1. Create the object detection model as given in the pipeline.config and convert is as a **custom** [Tensorflow Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
2. Check if **export** directory has a SavedModel (if resuming annotations) else, do a dummy training for 1 step and save the model as SavedModel in the **export** directory. The dummy one step training is required because, the [Tensorflow Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) does not allow **SavedModel** creation without having a checkpoint in **model** dir
3. Start [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving) and point it to **export** directory so that it can load updated models automatically for **predictions**
4. Perform assisted annotations in prodigy with predictions coming from **Tensorflow Serving**.
5. Use the annotations to train the model in the loop and optionally run evaluation, save the trained model as a **model.ckpt** in the **model** directory and **SavedModel** in **export dir**.
6. Run the **garbage collector**.
7. **Tensorflow Serving** automatically picks up the recent model present in the **export** and downs the previous model.
8. Repeat 4 and 5 until satisfied.

In a nutshell, **predictions** happen in **Tensorflow Serving** and the training happens parallely inside **Prodigy**. This structure ensures that, **predictions** can run parallely in a different hardware resource (CPU/GPU) and **training** and **evaluation** can run in another hardware resource(GPU/CPU). **GPU** for **training** and **evaluation** is highly recommended!

## Configuring the recipe:
This section explains how the [pipeline.config](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) and other arguments work in coherence for this recipe. This assumes that you have some prior knowledge on how to setup the [pipeline.config](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) for Tensorflow's Object Detection API.

* While starting this recipe first time for a new project, make sure to provide a `seed` [TF Record](https://www.tensorflow.org/tutorials/load_data/tf_records) containing **one training** example in **train_input_reader** config in the **pipeline.config**. This is required to do a dummy training for 1 step and save the model as SavedModel in the **export** directory. The dummy 1 step training is required because, the [Tensorflow Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) does not allow **SavedModel** creation without having a checkpoint in **model** directory. This **TF Record** can be created from a CSV file using the provided [create_tfrecord.py](../misc/create_tfrecord.py) script.
```python
train_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/train.record"
  }
}
```
However, if you are resuming annotations, you can skip the above, iff your **model** directory already has checkpoints from the previous runs.
* If you want to run the **evaluation** also in parallel(set by `run_eval` flag argument) you need to provide the **eval_input_reader** config in the **pipeline.config**.
```python
eval_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/validation.record"
  }
}
```
N number of samples are sampled from this **validation.record** (set by `eval_steps` argument) and evaluation is run on these examples. Supports all the [evaluation protocols](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md) supported by the Object Detection API

## Logging
* Set Prodigy logging level to `basic` to view detailed logs from this recipe. Optionally you can also set 10/20 for Tensorflow
* Optionally set [Tensorflow Logging](https://www.tensorflow.org/api_docs/python/tf/logging) to 10/20 if you want to see detailed Tensorflow logs. This is set by **tf_logging_level** argument

## Notes and Recommendations
* Set Prodigy logging level to `basic` to view detailed logs from this recipe. Optionally you can also set 10/20 for Tensorflow
* Object detection algorithms are extremely resource hungry! So, make sure that you run this recipe with **Tensorflow GPU**. However, you can choose to run **Tensorflow Serving** in **CPU** without much loss in performance.
* Point **TensorBoard** to **model** directory to view the training progress. The TensorBoard is really well populated. Especially with **evaluation** enabled.
* The recipe also supports all of the `data augmentations` provided by the Object Detection API out of the box. This can be enabled in the **pipeline_config**. This is especially useful if you are setting the **steps_per_epoch** argument to be more than the number of annotated examples.
* A custom **garbage collector** ensures that only recent N files/folders are stored in the **export** and *data** directory. This is specified by **temp_files_num** argument. The number of recent model checkpoints stored in **model** directory is governed by **max_checkpoints_num** argument.
* It is recommended to provide the `label_map_path` in the **pipeline.config** rather than passing it as an argument to the recipe
