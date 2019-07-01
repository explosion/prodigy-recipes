import os
import io
import copy
import grpc
import shutil
import functools
import numpy as np
import tensorflow as tf

from PIL import Image
from time import time

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from prodigy.components.loaders import get_stream
from prodigy.components.preprocess import fetch_images
from prodigy.core import recipe, recipe_args
from prodigy.util import log, b64_uri_to_bytes, split_string

from object_detection.utils import config_util, label_map_util
from object_detection.utils import dataset_util
from object_detection.builders import model_builder
from object_detection.model_lib import create_model_fn
from object_detection.inputs import create_train_input_fn
from object_detection.inputs import create_eval_input_fn
from object_detection.inputs import create_predict_input_fn
from object_detection.model_hparams import create_hparams


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
    label=("One or more comma-separated labels.\
        If not given inferred from labelmap",
           "option", "l", split_string, None, None),
    model_dir=("Path to save model checkpoints and Tensorboard events",
               "option", "md", str, None, os.path.join(".", "model_dir")),
    export_dir=("Path to save temporary SavedModels for Tensorflow Serving",
                "option", "ed", str, None, os.path.join(".", "export_dir")),
    data_dir=("Path to store temporary TFrecords used for training",
              "option", "dd", str, None, os.path.join(".", "data_dir")),
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
    tf_logging_level=("Log level for Tensorflow", "option",
                      "tl", int, (10, 20, 30, 40, 50), 40),
    api=recipe_args["api"],
    exclude=recipe_args["exclude"],
)
def image_trainmodel(dataset, source, config_path, ip, port, model_name,
                     label_map_path=None, label=None, model_dir="model_dir",
                     export_dir="export_dir", data_dir="data_dir",
                     steps_per_epoch=0, threshold=0.5, temp_files_num=5,
                     max_checkpoints_num=5, run_eval=False, eval_steps=50,
                     use_display_name=False, tf_logging_level=40, api=None,
                     exclude=None):
    tf.logging.set_verbosity(tf_logging_level)
    _create_dir(model_dir)
    _create_dir(export_dir)
    _create_dir(data_dir)
    log("Building the Tensorflow Object Detection API model")
    run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                        keep_checkpoint_max=max_checkpoints_num
                                        )
    odapi_configs = config_util.get_configs_from_pipeline_file(config_path)
    if label_map_path:
        log("Overriding label_map_path given in the odapi config file")
        odapi_configs["train_input_config"].label_map_path = label_map_path
        odapi_configs["eval_input_config"].label_map_path = label_map_path
    else:
        label_map_path = odapi_configs["train_input_config"].label_map_path

    # Set input reader config low to make sure you don't hit memory errors
    train_input_config = odapi_configs["train_input_config"]
    train_input_config.shuffle = False
    train_input_config.num_readers = 1
    train_input_config.num_parallel_batches = 1
    train_input_config.num_prefetch_batches = -1  # autotune
    train_input_config.queue_capacity = 2
    train_input_config.min_after_dequeue = 1
    train_input_config.read_block_length = 10
    train_input_config.prefetch_size = 2
    train_input_config.num_parallel_map_calls = 2

    # key class names
    reverse_class_mapping_dict = label_map_util.get_label_map_dict(
        label_map_path=label_map_path,
        use_display_name=use_display_name)
    if label is None:
        label = [k for k in reverse_class_mapping_dict.keys()]
    # key int
    class_mapping_dict = {v: k for k, v in reverse_class_mapping_dict.items()}

    detection_model_fn = functools.partial(model_builder.build,
                                           model_config=odapi_configs["model"])
    model_func = create_model_fn(detection_model_fn,
                                 hparams=create_hparams(None),
                                 configs=odapi_configs, use_tpu=False,
                                 postprocess_on_cpu=False)
    estimator = tf.estimator.Estimator(model_fn=model_func, config=run_config)
    if estimator.latest_checkpoint() is None:
        log("Running a single dummy training step!\
        else saving SavedModel for Tensorflow Serving does not work")
        train_input_config = odapi_configs["train_input_config"]
        train_input_fn = create_train_input_fn(
            train_config=odapi_configs["train_config"],
            model_config=odapi_configs["model"],
            train_input_config=train_input_config)
        estimator.train(input_fn=train_input_fn,
                        steps=1)
        _export_saved_model(export_dir, estimator, odapi_configs)
    log("Make sure to start Tensorflow Serving before opening Prodigy")
    log("Training and evaluation (if enabled) can be monitored by \
        pointing Tensorboard to {} directory".format(model_dir))

    stream = get_stream(source, api=api, loader="images", input_key="image")
    stream = fetch_images(stream)
    update_fn = functools.partial(
        update_odapi_model, estimator=estimator,
        data_dir=data_dir,
        reverse_class_mapping_dict=reverse_class_mapping_dict,
        odapi_configs=odapi_configs,
        steps_per_epoch=steps_per_epoch,
        export_dir=export_dir, run_eval=run_eval,
        eval_steps=eval_steps,
        temp_files_num=temp_files_num)

    return {
        "view_id": "image_manual",
        "dataset": dataset,
        "stream": get_image_stream(stream, class_mapping_dict,
                                   ip, port, model_name, float(threshold)),
        "exclude": exclude,
        "update": update_fn,
        "progress": lambda *args, **kwargs: 0,
        'config': {
            'label': ', '.join(label) if label is not None else 'all',
            'labels': label,       # Selectable label options,
        }
    }


def get_image_stream(stream, class_mapping_dict, ip, port, model_name, thresh):
    """Function that gets the image stream with bounding box information

    Arguments:
        stream (iterable): input image image stream
        class_mapping_dict (dict): with key as int and value as class name
        ip (str): tensorflow serving IP
        port (str): tensorflow serving port
        model_name (str): model name in tensorflow serving
        thresh (float): score threshold for predictions

    Returns:
        A generator that constantly yields a prodigy task
    """
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
        log("Using threshold {}, got {} predictions for file {}".format(
            thresh, len(eg["spans"]), eg["meta"]["file"]))
        task = copy.deepcopy(eg)
        yield task


def update_odapi_model(tasks, estimator, data_dir, reverse_class_mapping_dict,
                       odapi_configs, steps_per_epoch, export_dir, run_eval,
                       eval_steps, temp_files_num):
    """Update the object detection api model with annotations from prodigy

    Arguments:
        tasks (iterable): prodigy's tasks
        estimator (tf.estimator.Estimator): detection model as tf estimator
        data_dir (str): directory to store temp train TF-Records
        reverse_class_mapping_dict (dict): key as class name and value as int
        odapi_configs (dict): Object detection api pipeline.config object
        steps_per_epoch (int): Number of training steps.
        export_dir (str): directory to export temp SavedModels for TF serving
        run_eval (bool): Whether to run evaluation
        eval_steps (int): Number of steps for evaluations
        temp_files_num (int): Number of recent files/folders to keep in export\
        and data directories

    Returns:
        None if run_eval is False else evaluation loss (float)
    """
    train_data_name = "{}_train.record".format(int(time()))
    _write_tf_record(tasks=tasks,
                     output_file=os.path.join(data_dir,
                                              train_data_name),
                     reverse_class_mapping_dict=reverse_class_mapping_dict
                     )
    train_input_config = odapi_configs["train_input_config"]
    # delete existing input paths
    old_input_paths = train_input_config.tf_record_input_reader.input_path
    for i in range(len(old_input_paths)):
        del train_input_config.tf_record_input_reader.input_path[i]
    train_input_config.tf_record_input_reader.input_path.append(
        os.path.join(data_dir,
                     train_data_name
                     ))
    train_input_fn = create_train_input_fn(
        train_config=odapi_configs["train_config"],
        model_config=odapi_configs["model"],
        train_input_config=train_input_config)
    train_steps = steps_per_epoch
    if train_steps == 0:
        train_steps = len(tasks)
    log("Training for {} steps".format(train_steps))
    estimator.train(input_fn=train_input_fn,
                    steps=train_steps)
    _export_saved_model(export_dir, estimator, odapi_configs)
    # Keep only recent temp_files_num in temp dirs
    _remove_garbage(folder=export_dir,
                    max_num_to_keep=temp_files_num,
                    garbage_type="folder",
                    filter_string=None)

    _remove_garbage(folder=data_dir,
                    max_num_to_keep=temp_files_num,
                    garbage_type="file",
                    filter_string=".record")
    if run_eval:
        log("Running evaluation for {} steps".format(eval_steps))
        eval_input_config = odapi_configs["eval_input_config"]
        eval_input_config.shuffle = False
        eval_input_config.num_readers = 1
        eval_input_config.num_parallel_batches = 1
        eval_input_config.num_prefetch_batches = -1  # autotune
        eval_input_config.queue_capacity = 2
        eval_input_config.min_after_dequeue = 1
        eval_input_config.read_block_length = 10
        eval_input_config.prefetch_size = 2
        eval_input_config.num_parallel_map_calls = 2
        eval_input_function = create_eval_input_fn(
            eval_config=odapi_configs["eval_config"],
            eval_input_config=eval_input_config,
            model_config=odapi_configs["model"])
        eval_dict = estimator.evaluate(input_fn=eval_input_function,
                                       steps=eval_steps)
        return eval_dict["loss"]
    else:
        return None


def get_predictions(single_stream, class_mapping_dict, ip, port, model_name):
    """Gets predictions for a single image using Tensorflow serving

    Arguments:
        single_stream (dict): A single prodigy stream
        class_mapping_dict (dict): with key as int and value as class name
        ip (str): tensorflow serving IP
        port (str): tensorflow serving port
        model_name (str): model name in tensorflow serving

    Returns:
        A tuple containing numpy arrays:
        (class_ids, class_names, scores, boxes)
    """
    image_byte_stream = b64_uri_to_bytes(single_stream["image"])
    encoded_image_io = io.BytesIO(image_byte_stream)
    image = Image.open(encoded_image_io)
    width, height = image.size
    filename = str(single_stream["meta"]["file"])
    file_extension = filename.split(".")[1].lower()
    if file_extension == "png":
        image_format = b'png'
    elif file_extension in ("jpg", "jpeg"):
        image_format = b'jpg'
    else:
        log("Only 'png', 'jpeg' or 'jpg' files are supported by ODAPI.\
         Got {}. Thus treating it as `jpg` file.\
          Might cause errors".format(file_extension)
            )
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
    """Private function which exports a SavedModel from estimator
    Arguments:
        export_dir (str): directory to export temp SavedModels for TF serving
        estimator (tf.estimator.Estimator): detection model as tf estimator
        odapi_configs (dict): Object detection api pipeline.config object

    Returns:
        None
    """
    log("Exporting the model as SavedModel in {}".format(export_dir))
    # Just a placeholder
    pred_input_config = odapi_configs["eval_input_config"]
    predict_input_fn = create_predict_input_fn(odapi_configs["model"],
                                               pred_input_config)
    estimator.export_saved_model(export_dir_base=export_dir,
                                 serving_input_receiver_fn=predict_input_fn)
    log("Exported SavedModel!")


def _write_tf_record(tasks, output_file, reverse_class_mapping_dict):
    """Private function which writes training TF-Record file

    Arguments:
        tasks (iterable): prodigy's tasks
        output_file (str): output TF-Record filename
        reverse_class_mapping_dict (dict): key as class name and value as int

    Returns:
        None
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    for task in tasks:
        if task['answer'] == 'accept':
            tf_example = create_a_tf_example(task, reverse_class_mapping_dict)
            writer.write(tf_example.SerializeToString())
        else:
            continue
    writer.close()
    log("Successfully written {} annotations as TFRecords".format(len(tasks)))


def _create_dir(path):
    """A private function which creates a directory if it does not exists

    Arguments:
        path (str): Directory path

    Returns:
        None
    """
    if not os.path.isdir(path):
        log("Creating a directory {}".format(path))
        os.mkdir(path)
    else:
        log("Directory {} already  exists".format(path))


def get_span(prediction, pil_image, hidden=True):
    """Function which returns a prodigy span

    Arguments:
        prediction (iterable): containing one class_id, name, prob, box
        pil_image (pil.Image): A PIL image
        hidden (bool)

    Returns:
        A span (dict) with following keys:
        score, label, label_id, points, hidden
    """
    class_id, name, prob, box = prediction
    name = str(name, "utf8") if not isinstance(name, str) else name
    image_width = pil_image.width
    image_height = pil_image.height
    ymin, xmin, ymax, xmax = box
    # un-normalize the coordinates
    xmin = xmin*image_width
    xmax = xmax*image_width
    ymin = ymin*image_height
    ymax = ymax*image_height

    box_width = abs(xmax - xmin)
    box_height = abs(ymax - ymin)

    rel_points = [
        [xmin, ymin],
        [xmin, ymin+box_height],
        [xmin+box_width, ymin+box_height],
        [xmin+box_width, ymin]
    ]
    return {
        "score": prob,
        "label": name,
        "label_id": int(class_id),
        "points": rel_points,
        "hidden": hidden,
    }


def tf_odapi_client(data, ip, port, model_name,
                    signature_name, input_name, timeout=300):
    """Client for using Tensorflow Serving with Tensorflow Object Detection API

    Arguments:
        data (np.ndarray/bytes): A numpy array of data or bytes. No Default
        ip (str): IP address of tensorflow serving. No Default
        port (str/int): Port of tensorflow serving. No Default
        model_name (str): Model name. No Default
        signature_name (str): Signature name. No Default
        input_name (str): Input tensor name. No Default
        timeout (str): timeout for API call. Default 300 secs

    returns:
        a tuple containing numpy arrays of (boxes, classes, scores)
    """
    result = generic_tf_serving_client(data, ip, port,
                                       model_name, signature_name,
                                       input_name, timeout
                                       )
    # boxes are ymin.xmin,ymax,xmax
    boxes = np.array(result.outputs['detection_boxes'].float_val)
    classes = np.array(result.outputs['detection_classes'].float_val)
    scores = np.array(result.outputs['detection_scores'].float_val)
    boxes = boxes.reshape((len(scores), 4))
    classes = np.squeeze(classes.astype(np.int32))
    scores = np.squeeze(scores)

    return (boxes, classes, scores)


def generic_tf_serving_client(data, ip, port, model_name,
                              signature_name, input_name, timeout=300):
    """A generic tensorflow serving client that predicts using given data

    Arguments:
        data (np.ndarray/bytes): A numpy array of data or bytes. No Default
        ip (str): IP address of tensorflow serving. No Default
        port (str/int): Port of tensorflow serving. No Default
        model_name (str): Model name. No Default
        signature_name (str): Signature name. No Default
        input_name (str): Input tensor name. No Default
        timeout (str): timeout for API call. Default 300 secs

    returns:
        Prediction protobuf
    """
    start_time = time()
    assert isinstance(data, (np.ndarray, bytes)), \
        "data must be a numpy array or bytes but got {}".format(type(data))
    channel = grpc.insecure_channel('{}:{}'.format(ip, port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    request.inputs['{}'.format(input_name)
                   ].CopyFrom(tf.contrib.util.make_tensor_proto(
                       data,
                   ))
    result = stub.Predict(request, timeout)
    log("time taken for prediction using model {} \
    version {} is :{} secs".format(
        str(result.model_spec.name), result.model_spec.version.value,
        time()-start_time))
    return result


def create_a_tf_example(single_stream, reverse_class_mapping_dict):
    """Function to create a single training Tf.Example object

    Arguments:
        single_stream (dict): A single prodigy stream
        reverse_class_mapping_dict (dict): key as class name and value as int

    Returns:
        A single training tf.Example compatible with object detection API
    """
    image_byte_stream = b64_uri_to_bytes(single_stream["image"])
    encoded_image_io = io.BytesIO(image_byte_stream)
    image = Image.open(encoded_image_io)
    width, height = image.size
    filename = str(single_stream["meta"]["file"])
    file_extension = filename.split(".")[1].lower()
    if file_extension == "png":
        image_format = b'png'
    elif file_extension in ("jpg", "jpeg"):
        image_format = b'jpg'
    else:
        log("Only 'png', 'jpeg' or 'jpg' files are supported by ODAPI.\
         Got {}. Thus treating it as `jpg` file.\
          Might cause errors").format(file_extension)
        image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    filename = filename.encode("utf-8")
    for span in single_stream["spans"]:
        points = np.array(span["points"])
        xmin, ymin = np.amin(points, axis=0)
        xmax, ymax = np.amax(points, axis=0)
        # points need to be normalized
        xmin = xmin/width
        ymin = ymin/height
        xmax = xmax/width
        ymax = ymax/height
        assert xmin < xmax
        assert ymin < ymax
        # Clip bounding boxes that go outside the image
        if xmin < 0:
            xmin = 0
        if xmax > width:
            xmax = width - 1
        if ymin < 0:
            ymin = 0
        if ymax > height:
            ymax = height - 1
        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        classes_text.append(span["label"].encode("utf-8"))
        classes.append(int(reverse_class_mapping_dict[span["label"]]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(image_byte_stream),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text':
        dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def _remove_garbage(folder, max_num_to_keep, garbage_type,
                    filter_string=None):
    """Private function which keeps only max_num_to_keep files/folders
    in a given directory

    Arguments:
        folder (str): Folder to monitor
        max_num_to_keep (int): maximum number of recent files/folders to keep
        garbage_type (str): one of ('file' or 'folder').
        filer (str): optional pattern to look for. Default None

    Returns:
        None
    """
    contents = [os.path.join(folder, f) for f in os.listdir(folder)]
    if garbage_type.lower() == "file":
        contents = list(filter(lambda x: os.path.isfile(x), contents))
    elif garbage_type.lower() == "folder":
        contents = list(
            filter(lambda x: os.path.isdir(x) and "temp" not in str(x),
                   contents))
    else:
        raise ValueError("garbage_type must be one of 'file', 'folder'")
    if filter_string:
        contents = list(filter(lambda x: filter_string in os.path.basename(x),
                               contents))
    if len(contents) > max_num_to_keep:
        recent_n_contents = sorted(contents)[::-1][:max_num_to_keep]
        contents_to_delete = list(set(contents) - set(recent_n_contents))
        for content_to_delete in contents_to_delete:
            if garbage_type == "file":
                os.remove(content_to_delete)
            elif garbage_type == "folder":
                shutil.rmtree(content_to_delete, ignore_errors=True)
