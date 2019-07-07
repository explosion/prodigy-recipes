# coding: utf8
from __future__ import unicode_literals, print_function

import tensorflow as tf
import numpy as np
import copy
import io
from PIL import Image
from time import time

from prodigy.components.loaders import get_stream
from prodigy.components.preprocess import fetch_images
from prodigy.core import recipe, recipe_args
from prodigy.util import log, b64_uri_to_bytes

from object_detection.utils import label_map_util

detection_graph = None
sess = None


@recipe(
    "image.frozenmodel",
    dataset=recipe_args["dataset"],
    frozen_model_path=("Path to frozen_model.pb", "positional", None, str),
    label_map_path=("Path to label_map.pbtxt", "positional", None, str),
    source=recipe_args["source"],
    threshold=("Score threshold", "option", "t", float, None, 0.5),
    api=recipe_args["api"],
    exclude=recipe_args["exclude"],
    use_display_name=("Whether to use display_name in label_map.pbtxt",
                      "flag", "D", bool),
)
def image_tfodapimodel(dataset,
                       frozen_model_path,
                       label_map_path,
                       source=None,
                       threshold=0.5,
                       api=None,
                       exclude=None,
                       use_display_name=False
                       ):
    log("RECIPE: Starting recipe image.tfodapimodel", locals())
    log("RECIPE: Loading frozen model")
    global detection_graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    global sess
    sess = tf.Session(graph=detection_graph)
    log("RECIPE: Loaded frozen model")
    # key class names
    reverse_class_mapping_dict = label_map_util.get_label_map_dict(
        label_map_path=label_map_path,
        use_display_name=use_display_name
    )
    # key int
    class_mapping_dict = {v: k for k, v in reverse_class_mapping_dict.items()}
    stream = get_stream(source, api=api, loader="images", input_key="image")
    stream = fetch_images(stream)

    return {
        "view_id": "image_manual",
        "dataset": dataset,
        "stream": get_image_stream(stream, class_mapping_dict,
                                   float(threshold)),
        "exclude": exclude,
        "on_exit": free_graph,
    }


def get_image_stream(stream, class_mapping_dict, thresh):
    """Function that gets the image stream with bounding box information

    Arguments:
        stream (iterable): input image image stream
        class_mapping_dict (dict): with key as int and value as class name
        thresh (float): score threshold for predictions

    Returns:
        A generator that constantly yields a prodigy task
    """
    for eg in stream:
        if not eg["image"].startswith("data"):
            msg = "Expected base64-encoded data URI, but got: '{}'."
            raise ValueError(msg.format(eg["image"][:100]))

        pil_image = Image.open(io.BytesIO(b64_uri_to_bytes(eg["image"])))
        pil_image = preprocess_pil_image(pil_image)
        np_image = np.array(pil_image)
        predictions = get_predictions(np_image, class_mapping_dict)
        eg["width"] = pil_image.width
        eg["height"] = pil_image.height
        eg["spans"] = [get_span(pred, pil_image) for pred in
                       zip(*predictions) if pred[2] >= thresh]
        task = copy.deepcopy(eg)
        yield task


def preprocess_pil_image(pil_img, color_mode='rgb', target_size=None):
    """Preprocesses the PIL image

    Arguments
        img: PIL Image
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    Returns
        Preprocessed PIL image
    """
    if color_mode == 'grayscale':
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
    elif color_mode == 'rgba':
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
    elif color_mode == 'rgb':
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if pil_img.size != width_height_tuple:
            pil_img = pil_img.resize(width_height_tuple, Image.NEAREST)
    return pil_img


def get_predictions(numpy_image, class_mapping_dict):
    """Gets predictions for a single image using Frozen Model

    Arguments:
        numpy_image (np.ndarray): A single numpy image
        class_mapping_dict (dict): with key as int and value as class name

    Returns:
        A tuple containing numpy arrays:
        (class_ids, class_names, scores, boxes)
    """
    global detection_graph
    global sess
    image_tensor = detection_graph.get_tensor_by_name(
        'image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(numpy_image, axis=0)
    start_time = time()
    (boxes, scores, class_ids, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded}
    )
    log("time taken for image shape {} is {} secs".format(numpy_image.shape,
                                                          time()-start_time))
    boxes = np.squeeze(boxes)
    class_ids = np.squeeze(class_ids).astype(np.int32)
    class_names = np.array([class_mapping_dict[class_id]
                            for class_id in class_ids])
    scores = np.squeeze(scores)
    return (class_ids, class_names, scores, boxes)


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
    # boxes are in normalized coordinates
    # ymin, xmin, ymax, xmax
    ymin, xmin, ymax, xmax = box

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


def free_graph(ctrl):
    global detection_graph
    tf.reset_default_graph()
    global sess
    sess.close()
    del detection_graph
