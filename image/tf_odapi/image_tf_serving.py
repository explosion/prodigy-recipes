import tensorflow as tf
import grpc
import numpy as np
import copy
import io

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from PIL import Image

from prodigy.components.loaders import get_stream
from prodigy.components.preprocess import fetch_images
from prodigy.core import recipe, recipe_args
from prodigy.util import log, b64_uri_to_bytes

from object_detection.utils import label_map_util


@recipe(
    "image.servingmodel",
    dataset=recipe_args["dataset"],
    ip=("Tensorflow serving ip", "positional", None, str),
    port=("Tensorflow serving port", "positional", None, str),
    model_name=("Tensorflow serving model name", "positional", None, str),
    label_map_path=("Path to label_map.pbtxt", "positional", None, str),
    source=recipe_args["source"],
    threshold=("Score threshold", "option", "t", float, None, 0.5),
    api=recipe_args["api"],
    exclude=recipe_args["exclude"],
)
def image_servingmodel(dataset,
                       ip,
                       port,
                       model_name,
                       label_map_path,
                       source=None,
                       threshold=0.5,
                       api=None,
                       exclude=None
                       ):
    log("RECIPE: Starting recipe image.servingmodel", locals())

    # key class names
    reverse_class_mapping_dict = label_map_util.get_label_map_dict(
        label_map_path=label_map_path,
        use_display_name=True)
    # key int
    class_mapping_dict = {v: k for k, v in reverse_class_mapping_dict.items()}
    stream = get_stream(source, api=api, loader="images", input_key="image")
    stream = fetch_images(stream)

    return {
        "view_id": "image_manual",
        "dataset": dataset,
        "stream": get_image_stream(stream, class_mapping_dict,
                                   ip, port, model_name, float(threshold)),
        "exclude": exclude,
    }


def get_image_stream(stream, class_mapping_dict, ip, port, model_name, thresh):
    for eg in stream:
        if not eg["image"].startswith("data"):
            msg = "Expected base64-encoded data URI, but got: '{}'."
            raise ValueError(msg.format(eg["image"][:100]))

        pil_image = Image.open(io.BytesIO(b64_uri_to_bytes(eg["image"])))
        pil_image = preprocess_pil_image(pil_image)
        np_image = np.array(pil_image)
        predictions = get_predictions(np_image, class_mapping_dict,
                                      ip, port, model_name)
        eg["width"] = pil_image.width
        eg["height"] = pil_image.height
        eg["spans"] = [get_span(pred, pil_image)
                       for pred in zip(*predictions) if pred[2] >= thresh]
        task = copy.deepcopy(eg)
        yield task


def get_predictions(numpy_image, class_mapping_dict, ip, port, model_name):
    if len(numpy_image.shape) == 3:
        numpy_image = np.expand_dims(numpy_image, axis=0)
    boxes, class_ids, scores = _tf_odapi_client(numpy_image,
                                                ip,
                                                port, model_name)
    class_names = np.array([class_mapping_dict[class_id]
                            for class_id in class_ids])
    return (class_ids, class_names, scores, boxes)


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


def _tf_odapi_client(image, ip, port, model_name,
                     signature_name="detection_signature", input_name="inputs",
                     timeout=300):
    result = _generic_tf_serving_client(image, ip, port,
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


def _generic_tf_serving_client(data, ip, port, model_name,
                               signature_name, input_name, timeout=300):
    """A generic tensorflow serving client that predicts using given data

    Arguments:
        data (np.ndarray): A numpy array of data. No Default
        ip (str): IP address of tensorflow serving. No Default
        port (str/int): Port of tensorflow serving. No Default
        model_name (str): Model name. No Default
        signature_name (str): Signature name. No Default
        input_name (str): Input tensor name. No Default
        timeout (str): timeout for API call. Default 300 secs

    returns:
        Prediction protobuf
    """
    assert isinstance(data, np.ndarray), \
        "data must be a numpy array but got {}".format(type(data))
    channel = grpc.insecure_channel('{}:{}'.format(ip, port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    request.inputs['{}'.format(input_name)
                   ].CopyFrom(tf.contrib.util.make_tensor_proto(
                       data,
                       shape=data.shape))
    result = stub.Predict(request, timeout)
    return result
