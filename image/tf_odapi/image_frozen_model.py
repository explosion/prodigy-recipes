# coding: utf8
from __future__ import unicode_literals, print_function

import tensorflow as tf
import numpy as np
import copy
import io
from PIL import Image

from prodigy.components.loaders import get_stream
from prodigy.components.preprocess import fetch_images
from prodigy.core import recipe, recipe_args
from prodigy.util import log, b64_uri_to_bytes

from object_detection.utils import label_map_util

detection_graph = None


@recipe(
    "image.frozenmodel",
    dataset=recipe_args["dataset"],
    frozen_model_path=("Path to frozen model", "positional", None, str),
    label_map_path=("Path to label_map.pbtxt", "positional", None, str),
    source=recipe_args["source"],
    api=recipe_args["api"],
    exclude=recipe_args["exclude"],
)
def image_tfodapimodel(dataset,
                       frozen_model_path,
                       label_map_path,
                       source=None,
                       api=None,
                       exclude=None
                       ):
    log("RECIPE: Starting recipe image.tfodapimodel", locals())

    def get_image_stream(stream, thresh=0.5):
        for eg in stream:
            if not eg["image"].startswith("data"):
                msg = "Expected base64-encoded data URI, but got: '{}'."
                raise ValueError(msg.format(eg["image"][:100]))

            pil_image = Image.open(io.BytesIO(b64_uri_to_bytes(eg["image"])))
            np_image = np.array(pil_image)
            predictions = get_predictions(np_image)
            eg["width"] = pil_image.width
            eg["height"] = pil_image.height
            eg["spans"] = [get_span(pred, pil_image) for pred in zip(*predictions) if pred[2] >= 0.5]
            task = copy.deepcopy(eg)
            yield task

    def get_predictions(numpy_image):
        global detection_graph
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                image_np_expanded = np.expand_dims(numpy_image, axis=0)
                (boxes, scores, class_ids, num) = sess.run(
                    [detection_boxes, detection_scores,
                     detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded}
                )
        # boxes are in normalized coordinates
        # ymin, xmin, ymax, xmax
        boxes = np.squeeze(boxes)
        class_ids = np.squeeze(class_ids).astype(np.int32)
        class_names = np.array([class_mapping_dict[class_id] for class_id in class_ids])
        scores = np.squeeze(scores)
        return (class_ids, class_names, scores, boxes)

    def get_span(prediction, pil_image, hidden=True):
        class_id, name, prob, box = prediction
        name = str(name, "utf8") if not isinstance(name, str) else name
        image_width = pil_image.width
        image_height = pil_image.height
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

    global detection_graph

    log("RECIPE: Loading frozen model")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    log("RECIPE: Loaded frozen model")
    # key class names
    reverse_class_mapping_dict = label_map_util.get_label_map_dict(label_map_path=label_map_path,
                                                                   use_display_name=True)
    # key int
    class_mapping_dict = {v: k for k, v in reverse_class_mapping_dict.items()}
    stream = get_stream(source, api=api, loader="images", input_key="image")
    stream = fetch_images(stream)

    def free_graph(ctrl):
        global detection_graph
        tf.reset_default_graph()
        del detection_graph

    return {
        "view_id": "image_manual",
        "dataset": dataset,
        "stream": get_image_stream(stream),
        "exclude": exclude,
        "on_exit": free_graph,
    }
