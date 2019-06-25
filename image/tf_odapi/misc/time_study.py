import tensorflow as tf
import os
from tqdm import tqdm
from time import time
import numpy as np
from argparse import RawTextHelpFormatter, ArgumentParser
import pandas as pd


def main(model_base_path, image, warm_up_itr, study_itr):
    print("Loading frozen graph...")
    frozen_model_path = os.path.join(model_base_path,
                                     "frozen_inference_graph.pb")
    st_time = time()
    image_np_expanded = np.expand_dims(image, axis=0)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("time taken for loading graph is {}".format(time()-st_time))

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
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
            print("warming up the device for {} iters".format(warm_up_itr))
            for _ in range(warm_up_itr):
                (boxes, scores, class_ids, num) = sess.run(
                    [detection_boxes, detection_scores,
                     detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded}
                )
            print("Running time study for model {} for {} iterations".format(
                os.path.basename(model_base_path),
                study_itr))
            start_time = time()
            for _ in tqdm(range(study_itr)):
                (boxes, scores, class_ids, num) = sess.run(
                    [detection_boxes, detection_scores,
                     detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded}
                )
            end_time = time()
            total_time = end_time - start_time
            avg_time = total_time/study_itr
            print("Total time for {} iters is {} secs".format(study_itr,
                                                              total_time))
            print("Average time per iter is {}".format(avg_time))

    tf.reset_default_graph()
    del detection_graph
    print("Graph cleared!")
    return (total_time, avg_time)


if __name__ == "__main__":
    parser = ArgumentParser(description='TF-ODAPI time study',
                            formatter_class=RawTextHelpFormatter
                            )
    parser.add_argument("base_dir",
                        help="path to directory containing model sub dirs",
                        type=str
                        )
    parser.add_argument("image_path",
                        help="path to image",
                        type=str
                        )
    parser.add_argument("output_csv_file",
                        help="path to output csv file",
                        type=str
                        )
    parser.add_argument("--device", "-d",
                        help="GPU device to run. Default '0'. Use -1 for CPU",
                        type=str, metavar='', default="0"
                        )
    parser.add_argument("--warm_up_itr", "-w",
                        help="Number of warmup iters. Default 5",
                        type=int, metavar='', default=5
                        )

    parser.add_argument("--study_itr", "-s",
                        help="Number of iters for time study. Default 100",
                        type=int, metavar='', default=100
                        )
    args = parser.parse_args()
    total_times = []
    average_times = []
    model_names = []

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    dirs = os.listdir(args.base_dir)
    dirs = list(filter(lambda x: True if ".tar.gz" not in x else False, dirs))
    print("This time study is performed with image size of (600,1000,3)")
    print("Found {} models in the given directory".format(len(dirs)))
    pil_image = tf.keras.preprocessing.image.load_img(args.image_path,
                                                      target_size=(600, 1000))
    np_image = np.array(pil_image)
    for i, dir in enumerate(sorted(dirs)):
        if "quantized" in dir:
            print("{}. Skipping {} because it uses TF-Lite"
                  .format(i+1, os.path.basename(dir)))
            continue
        print("\n##########################################################\n")
        print("{}. Starting time study for {}".format(i+1,
                                                      os.path.basename(dir)))
        model_base_path = os.path.join(args.base_dir, dir)
        total_time, avg_time = main(model_base_path, np_image,
                                    warm_up_itr=args.warm_up_itr,
                                    study_itr=args.study_itr)
        model_names.append(os.path.basename(dir))
        total_times.append(total_time)
        average_times.append(avg_time)

        print("\n##########################################################\n")

    df = pd.DataFrame({
        "model_name": model_names,
        "warm_up_itr": args.warm_up_itr,
        "study_itr": args.study_itr,
        "total_time": total_times,
        "average_time": average_times
    })
    df.to_csv(args.output_csv_file)
