# Tensorflow Detection model zoo

The document contains **Speed and Accuracy trade-off** between different models given in the original [detection_model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). The models will be slower compared to the time reported in the **detection_model_zoo** for most cases because, the device informations are cleared when freezing the graph for portability reasons. Thus, the optimal GPU/CPU placement for the Ops are lost! Look at this [issue](https://github.com/tensorflow/models/issues/3270) for more details. Thus, the idea of this document is to provide the inference speed and **Mean Average Precision** mAP for the frozen models given in the **detection_model_zoo**. The mAP values are taken as such from the original **detection_model_zoo**

## Study Details

*   **Input image size**: (600, 1000, 3). This size is chosen because, The original [Faster-RCNN paper](https://arxiv.org/abs/1506.01497) resizes the images to this shape.
*   **GPU used**: Nvidia GTX 1060 with 6GB memory
*   **CPU used**: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
*   The CPU version of tensorflow was **not** compiled with AVX2 and other available optimizations (`pip install tensorflow`). Using optimized tensorflow CPU binary should increase the speed further. See this [blog](https://www.anaconda.com/tensorflow-in-anaconda/) for more details.
*   The script used for this study is [time_study.py](./misc/time_study.py)

## Models trained on MS-COCO Dataset

| S.No |  Model Name                                                   | GPU time (ms) | CPU time (ms) | mAP  |
|------|---------------------------------------------------------------|---------------|---------------|------|
| 1    | faster_rcnn_inception_resnet_v2_atrous_coco                   | 1093          | 14685         | 37   |
| 2    | faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco      | 560           | 6333          |      |
| 3    | faster_rcnn_inception_v2_coco                                 | 92            | 670           | 28   |
| 4    | faster_rcnn_resnet101_coco                                    | 188           | 2697          | 32   |
| 5    | faster_rcnn_resnet101_lowproposals_coco                       | 171           | 1655          |      |
| 6    | faster_rcnn_resnet50_coco                                     | 178           | 1731          | 30   |
| 7    | faster_rcnn_resnet50_lowproposals_coco                        | 121           | 994           |      |
| 8    | rfcn_resnet101_coco                                           | 180           | 2071          | 30   |
| 9    | ssd_inception_v2_coco                                         | 34            | 85            | 24   |
| 10   | ssd_mobilenet_v1_0.75_coco                                    | 23            | 32            | 18   |
| 11   | ssd_mobilenet_v1_coco                                         | 20            | 41            | 21   |
| 12   | ssd_mobilenet_v1_fpn_coco                                     | 107           | 909           | 32   |
| 13   | ssd_mobilenet_v1_ppn_coco                                     | 20            | 39            | 20   |
| 14   | ssd_mobilenet_v2_coco                                         | 25            | 56            | 22   |
| 15   | ssd_resnet50_fpn_coco                                         | 136           | 1322          | 35   |
| 16   | ssdlite_mobilenet_v2_coco                                     | 21            | 43            | 22   |


## NOTES:
*   The input image size will **not** affect the inference speed (until a certain range of input shapes). This is because, the resizing function is baked into the Tensorflow graph. The time taken for resizing should be insignificant compared to the other Ops's time. However, take this with a grain of salt as this is valid only until certain range of input shapes.
*   Batching the input images doesn't seem to increase the inference performance. As in, the inference time is linear with the batch size. See this [issue](https://github.com/tensorflow/models/issues/4266) for more details
