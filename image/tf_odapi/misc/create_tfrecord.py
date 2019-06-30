"""
Create TFRecords for Tensorflow's Object Detection API

`python create_tfrecord.py --help` to know how to use this script

Notes on the input csv file:
- The file must contain one bounding box per row.
- Must contain the following column names: path, xmin, ymin, xmax, ymax, label
Where: `path` is, path to the image and `label` is, class name of object
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import io
import os
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils.label_map_util import get_label_map_dict
from argparse import RawTextHelpFormatter, ArgumentParser
from collections import namedtuple
from tqdm import tqdm


def _split(df, group):
    """Private function"""
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in
            zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, label_map_dict):
    """Creates one tf example"""
    if not os.path.isfile(group.filename):
        raise FileNotFoundError("{} file does not exists".format(
            group.filename))
    with tf.gfile.GFile('{}'.format(group.filename), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    img_file_format = os.path.splitext(os.path.basename(
        group.filename))[1].lower()
    if img_file_format == '.png':
        image_format = b'png'
    elif img_file_format == '.jpg' or img_file_format == '.jpeg':
        image_format = b'jpg'
    else:
        raise ValueError("The image must be of format .jpg, .jpeg or .png")
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        assert row['xmin'] < row['xmax'], \
            "{} xmin is not less than {} xmax".format(row['xmin'], row['xmax'])
        assert row['ymin'] < row['ymax'], \
            "{} ymin is not less than {} ymax".format(row['ymin'], row['ymax'])
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        # Clip bounding boxes that go outside the image
        if xmin < 0:
            xmin = 0
        if xmax > width:
            xmax = width - 1
        if ymin < 0:
            ymin = 0
        if ymax > height:
            ymax = height - 1
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes_text.append(row['label'].encode('utf8'))
        classes.append(label_map_dict[row['label']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
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


def main(args):
    writer = tf.python_io.TFRecordWriter(args.output_path)
    examples = pd.read_csv(args.csv_input)
    if args.base_path:
        examples.path = [os.path.join(args.base_path, p)
                         for p in examples.path]
    grouped = _split(examples, 'path')
    label_map_dict = get_label_map_dict(label_map_path=args.label_map_path)
    for group in tqdm(grouped):
        tf_example = create_tf_example(group, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(args.output_path))


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Create TFRecords to use with TF Object Detection API.",
        formatter_class=RawTextHelpFormatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument("--csv_input",
                          help="Path to input csv file",
                          type=str, metavar='', default=None)
    required.add_argument("--label_map_path", help="Path to label_map.pbtxt",
                          type=str, metavar='', default=None)
    required.add_argument("--output_path", help="Path to output.record",
                          type=str, metavar='', default=None)
    optional.add_argument("--base_path",
                          help="Base path to the images.\
    Supply this path if your path field idoes not contain the complete path",
                          type=str, metavar='', default=None)
    parser._action_groups.append(optional)
    args = parser.parse_args()
    if not args.csv_input or not args.output_path or not args.label_map_path:
        parser.error("Compulsary arguments '--csv_input',\
                     '--output_path' and '--label_map_path' must be given")
    main(args)
