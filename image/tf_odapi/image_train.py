import os
import io
import copy
import functools
import numpy as np
import tensorflow as tf

from PIL import Image
from time import time

from prodigy.components.loaders import get_stream
from prodigy.components.preprocess import fetch_images
from prodigy.core import recipe, recipe_args
from prodigy.util import log, b64_uri_to_bytes

from object_detection.utils import config_util, label_map_util
from object_detection.utils import dataset_util
from object_detection.builders import model_builder
from object_detection.model_lib import create_model_fn
from object_detection.inputs import create_train_input_fn
from object_detection.inputs import create_eval_input_fn
from object_detection.inputs import create_predict_input_fn
from object_detection.model_hparams import create_hparams

from utils import get_span, tf_odapi_client, create_a_tf_example

estimator = None


@recipe(
    "image.trainmodel",
    dataset=recipe_args["dataset"],
    source=recipe_args["source"],
    config_path=("Path to tfodapi config file", "positional", None, str),
    ip=("Tensorflow serving ip", "positional", None, str),
    port=("Tensorflow serving port", "positional", None, str),
    model_name=("Tensorflow serving model name", "positional", None, str),
    label_map_path=("Labelmap.pbtxt path. Overrides the value given in \
    tfodapi config", "option", "lmp", float, None, None),
    model_dir=("Path to save model checkpoints and Tensorboard events",
               "option", "md", float, None, os.path.join(".", "model_dir")),
    export_dir=("Path to save temporary SavedModels for Tensorflow Serving",
                "option", "ed", float, None, os.path.join(".", "export_dir")),
    data_dir=("Path to store temporary TFrecords used for training",
              "option", "dd", float, None, os.path.join(".", "data_dir")),
    steps_per_epoch=("Number of training steps per epoch.\
    If 0, inferred automatically. If higher than the dataset size,\
    the dataset is looped over",
                     "option", "spe", int, None, 0),
    threshold=("Score threshold", "option", "t", float, None, 0.5),
    temp_files_num=("Number of recent temp files to keep",
                    "option", "tfn", int, None, 5),
    max_checkpoints_num=("Number of recent model checkpoints to keep",
                         "option", "mcn", int, None, 5),
    run_eval=("Whether to run evaluation. If enabling, eval_config\
    and eval_input_reader must be set in tfodapi config",
              "flag", "E", bool),
    eval_steps=("Number of steps for evaluation",
                "option", "es", int, None, 50),
    use_display_name=("Whether to use display_name in label_map.pbtxt",
                      "flag", "D", bool),
    api=recipe_args["api"],
    exclude=recipe_args["exclude"],
)
def image_trainmodel(dataset, source, config_path, ip, port, model_name,
                     label_map_path=None, model_dir="model_dir",
                     export_dir="export_dir", data_dir="data_dir",
                     steps_per_epoch=0, threshold=0.5, temp_files_num=5,
                     max_checkpoints_num=5, run_eval=False, eval_steps=50,
                     use_display_name=False, api=None, exclude=None
                     ):
    # key class names
    _create_dir(model_dir)
    _create_dir(export_dir)
    _create_dir(data_dir)
    reverse_class_mapping_dict = label_map_util.get_label_map_dict(
        label_map_path=label_map_path,
        use_display_name=use_display_name)
    # key int
    class_mapping_dict = {v: k for k, v in reverse_class_mapping_dict.items()}
    log("Building the Tensorflow Object Detection API model")
    run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                        keep_checkpoint_max=max_checkpoints_num
                                        )
    odapi_configs = config_util.get_configs_from_pipeline_file(config_path)
    if label_map_path:
        log("Overriding label_map_path given in the odapi config file")
        odapi_configs["train_input_config"].label_map_path = label_map_path
        odapi_configs["eval_input_config"].label_map_path = label_map_path

    detection_model_fn = functools.partial(model_builder.build,
                                           model_config=odapi_configs["model"])
    model_func = create_model_fn(detection_model_fn,
                                 hparams=create_hparams(None),
                                 configs=odapi_configs, use_tpu=False,
                                 postprocess_on_cpu=False)
    estimator = tf.estimator.Estimator(model_fn=model_func, config=run_config)
    _export_saved_model(export_dir, estimator, odapi_configs)
    log("Make sure to start Tensorflow Serving before opening Prodigy")

    def update_odapi_model(tasks):
        train_data_name = "{}_train.record".format(int(time()))
        _write_tf_record(tasks=tasks, output_file=os.path.join(data_dir,
                                                               train_data_name
                                                               ))
        temp_configs = copy.deepcopy(odapi_configs)
        train_input_config = temp_configs["train_input_config"]
        # delete existing input paths
        old_input_paths = train_input_config.tf_record_input_reader.input_path
        for i in range(len(old_input_paths)):
            del train_input_config.tf_record_input_reader.input_path[i]
        train_input_config.tf_record_input_reader.input_path.append(
            os.path.join(data_dir,
                         train_data_name
                         ))
        train_input_fn = create_train_input_fn(
            train_config=temp_configs["train_config"],
            model_config=temp_configs["model"],
            train_input_config=train_input_config)
        log("Training for {} steps".format(steps_per_epoch))
        train_steps = steps_per_epoch
        if train_steps == 0:
            train_steps = len(tasks)
        estimator.train(input_fn=train_input_fn,
                        steps=train_steps)
        _export_saved_model(export_dir, estimator, temp_configs)
        if run_eval:
            eval_input_config = temp_configs["eval_input_config"]
            eval_input_function = create_eval_input_fn(
                eval_config=temp_configs["eval_config"],
                eval_input_config=eval_input_config,
                model_config=temp_configs["model"])
            eval_dict = estimator.evaluate(input_fn=eval_input_function,
                                           steps=eval_steps)
            return eval_dict["loss"]
        else:
            return None

    stream = get_stream(source, api=api, loader="images", input_key="image")
    stream = fetch_images(stream)
    return {
        "view_id": "image_manual",
        "dataset": dataset,
        "stream": get_image_stream(stream, class_mapping_dict,
                                   ip, port, model_name, float(threshold)),
        "exclude": exclude,
        "update": update_odapi_model,
        "progress": lambda *args, **kwargs: 0
    }


def get_image_stream(stream, class_mapping_dict, ip, port, model_name, thresh):
    for eg in stream:
        if not eg["image"].startswith("data"):
            msg = "Expected base64-encoded data URI, but got: '{}'."
            raise ValueError(msg.format(eg["image"][:100]))

        pil_image = Image.open(io.BytesIO(b64_uri_to_bytes(eg["image"])))
        predictions = get_predictions(eg, class_mapping_dict,
                                      ip, port, model_name)
        eg["width"] = pil_image.width
        eg["height"] = pil_image.height
        eg["spans"] = [get_span(pred, pil_image)
                       for pred in zip(*predictions) if pred[2] >= thresh]
        task = copy.deepcopy(eg)
        yield task


def get_predictions(single_stream, class_mapping_dict, ip, port, model_name):
    image_byte_stream = b64_uri_to_bytes(single_stream["image"])
    encoded_image_io = io.BytesIO(image_byte_stream)
    image = Image.open(encoded_image_io)
    width, height = image.size
    filename = str(single_stream["meta"]["file"])
    file_extension = filename.split(".").lower()
    if file_extension == "png":
        image_format = b'png'
    elif file_extension in ("jpg", "jpeg"):
        image_format = b'jpg'
    else:
        log("Only 'png', 'jpeg' or 'jpg' files are supported by ODAPI.\
         Got {}. Thus treating it as `jpg` file.\
          Might cause errors").format(file_extension)
        image_format = b'jpg'

    filename = filename.encode("utf-8")
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(image_byte_stream),
        'image/format': dataset_util.bytes_feature(image_format),
    }))

    boxes, class_ids, scores = tf_odapi_client(tf_example.SerializeToString(),
                                               ip, port, model_name,
                                               "serving_default",
                                               input_name="serialized_example",
                                               timeout=300
                                               )
    class_names = np.array([class_mapping_dict[class_id]
                            for class_id in class_ids])
    return (class_ids, class_names, scores, boxes)


def _export_saved_model(export_dir, estimator, odapi_configs):
    log("Exporting the model as SavedModel in {}").format(export_dir)
    # Just a placeholder
    pred_input_config = odapi_configs["eval_input_config"]
    predict_input_fn = create_predict_input_fn(odapi_configs["model"],
                                               pred_input_config)
    estimator.export_saved_model(export_dir_base=export_dir,
                                 serving_input_receiver_fn=predict_input_fn)
    log("Exported SavedModel!")


def _write_tf_record(tasks, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    for task in tasks:
        tf_example = create_a_tf_example(task)
        writer.write(tf_example.SerializeToString())
    writer.close()
    log("Successfully written {} annotations as TFRecords".format(len(tasks)))


def _create_dir(path):
    if not os.path.isdir(path):
        log("Creating a directory {}".format(path))
        os.mkdir(path)
    else:
        log("Directory {} already  exists".format(path))
