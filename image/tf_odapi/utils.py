import io
import grpc
import shutil
import numpy as np
import tensorflow as tf

from PIL import Image
from time import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from object_detection.utils import dataset_util
from prodigy.util import log, b64_uri_to_bytes


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


def tf_odapi_client(data, ip, port, model_name,
                    signature_name="detection_signature", input_name="inputs",
                    timeout=300):
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
    start_time = time()
    result = generic_tf_serving_client(data, ip, port,
                                       model_name, signature_name,
                                       input_name, timeout
                                       )
    log("time taken for prediction is :{} secs".format(time()-start_time))
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
    return result


def create_a_tf_example(single_stream):
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

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    filename = filename.encode("utf-8")
    for span in single_stream["spans"]:
        # points are ordered counter-clockwise
        points = span["points"]
        # points need to be normalized
        xmin = points[0][0]/width
        ymin = points[0][1]/height
        xmax = points[2][0]/width
        ymax = points[2][1]/width
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
        ymaxs.append(ymaxs)
        classes_text.append(span["label"].encode("utf-8"))
        classes.append(int(span["label_id"]))

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
        'image/object/class/text': dataset_util.bytes_list_feature(
            classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def remove_garbage(dir, max_num):
    pass
