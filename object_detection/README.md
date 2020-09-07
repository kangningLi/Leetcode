# Training tricks
#### Bag of Freebies for Training Object Detection Neural Networks
- arxiv: [https://arxiv.org/abs/1902.04103](https://arxiv.org/abs/1902.04103)
#### Bag of Tricks for Image Classification with Convolutional Neural Networks
- arxiv: [https://arxiv.org/abs/1812.01187](https://arxiv.org/abs/1812.01187)

# Survey  

#### Imbalance Problems in Object Detection: A Review  

- arXiv: [https://arxiv.org/abs/1909.00169](https://arxiv.org/abs/1909.00169)
- github: [https://github.com/kemaloksuz/ObjectDetectionImbalance](https://github.com/kemaloksuz/ObjectDetectionImbalance)(Studies on imbalanced problems)  

#### Recent Advances in Deep Learning for Object Detection

- Intro: From 2013 (OverFeat) to 2019 (DetNAS)
- arXiv: [https://arxiv.org/abs/1908.03673](https://arxiv.org/abs/1908.03673)

#### A Survey of Deep Learning-based Object Detection

- Intro: From Fast R-CNN to NAS-FPN
- arXiv: [https://arxiv.org/abs/1907.09408](https://arxiv.org/abs/1907.09408)

#### Object Detection in 20 Years: A Survey

- arXiv: [https://arxiv.org/abs/1905.05055](https://arxiv.org/abs/1905.05055)

#### Recent Advances in Object Detection in the Age of Deep Convolutional Neural Networks

- arXiv: [https://arxiv.org/abs/1809.03193](https://arxiv.org/abs/1809.03193)

#### Deep Learning for Generic Object Detection: A Survey

- Intro: Submitted to IJCV 2018
- arXiv: [https://arxiv.org/abs/1809.02165](https://arxiv.org/abs/1809.02165)  

# Papers & Code

## R-CNN
#### Rich feature hierarchies for accurate object detection and semantic segmentation
- arXiv: [https://arxiv.org/abs/1311.2524](https://arxiv.org/abs/1311.2524)
- Author page: [http://www.rossgirshick.info/](http://www.rossgirshick.info/)
- Slide: [http://www.rossgirshick.info/](http://www.rossgirshick.info/)
- github: [https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)

## Fast R-CNN
- arXiv: [https://arxiv.org/abs/1504.08083](https://arxiv.org/abs/1504.08083)
- tutorial(with caffe): [http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf)
- code: [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
- code(coco branch):[https://github.com/rbgirshick/fast-rcnn/tree/coco](https://github.com/rbgirshick/fast-rcnn/tree/coco)
- code(MXnet): [https://github.com/ijkguo/mx-rcnn](https://github.com/ijkguo/mx-rcnn)
- code(torch): [https://github.com/mahyarnajibi/fast-rcnn-torch](https://github.com/mahyarnajibi/fast-rcnn-torch)
- code(tensorflow): [https://github.com/zplizzi/tensorflow-fast-rcnn](https://github.com/zplizzi/tensorflow-fast-rcnn)

#### A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection
- intro: CVPR 2017
- arXiv: [https://arxiv.org/abs/1704.03414](https://arxiv.org/abs/1704.03414)
- code(caffe): [https://github.com/xiaolonw/adversarial-frcnn](https://github.com/xiaolonw/adversarial-frcnn)

## Faster R-CNN
#### Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- intro: NIPS 2015
- arXiv: [https://arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497)
- slides: [https://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w05-FasterR-CNN.pdf](https://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w05-FasterR-CNN.pdf)
- github(official, matlab): [https://github.com/ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)
- github(caffee): [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- github(MXNet): [https://github.com/msracver/Deformable-ConvNets/tree/master/faster_rcnn](https://github.com/msracver/Deformable-ConvNets/tree/master/faster_rcnn)
- github(PyTorch--recommend): [https://github.com//jwyang/faster-rcnn.pytorch](https://github.com//jwyang/faster-rcnn.pytorch)
- github(torch): [https://github.com/andreaskoepf/faster-rcnn.torch](https://github.com/andreaskoepf/faster-rcnn.torch)
- github(tensorflow): [https://github.com/smallcorgi/Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)
- github(c++ demo): [https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus](https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus)
- github(c++): [https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev](https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev)

## YOLO

#### You Only Look Once: Unified, Real-Time Object Detection(YOLOV1)
- arxiv: [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)
- official_blog: [https://pjreddie.com/darknet/yolov1/](https://pjreddie.com/darknet/yolov1/)
- github: [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)
- slide: [https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p)
- github(tensorflow): [https://github.com/gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)
- github(caffee): [https://github.com/xingwangsfu/caffe-yolo](https://github.com/xingwangsfu/caffe-yolo)
- github(yolo-windows): [https://github.com/frischzenger/yolo-windows](https://github.com/frischzenger/yolo-windows)
- github(yolo-windows): [https://github.com/AlexeyAB/yolo-windows](https://github.com/AlexeyAB/yolo-windows)
- github(tensorflow-yolo): [https://github.com/nilboy/tensorflow-yolo](https://github.com/nilboy/tensorflow-yolo)

#### darkflow - translate darknet to tensorflow. Load trained weights, retrain/fine-tune them using tensorflow, export constant graph def to C++
- github: [https://github.com/Guanghan/darknet](https://github.com/Guanghan/darknet)

#### Start Training YOLO with Our Own Data
- blog: [http://guanghan.info/blog/en/my-works/train-yolo/](http://guanghan.info/blog/en/my-works/train-yolo/)
- github: [https://github.com/Guanghan/darknet](https://github.com/Guanghan/darknet)

#### YOLO: Core ML versus MPSNNGraph
- intro: Tiny YOLO for iOS implemented using CoreML but also using the new MPS graph API.
- blog: [http://machinethink.net/blog/yolo-coreml-versus-mps-graph/](http://machinethink.net/blog/yolo-coreml-versus-mps-graph/)
- github: [https://github.com/hollance/YOLO-CoreML-MPSNNGraph](https://github.com/hollance/YOLO-CoreML-MPSNNGraph)

#### TensorFlow YOLO object detection on Android
- intro: Real-time object detection on Android using the YOLO network with TensorFlow
- github: [https://github.com/natanielruiz/android-yolo](https://github.com/natanielruiz/android-yolo)

#### Computer Vision in iOS – Object Detection
- github: [https://github.com/r4ghu/iOS-CoreML-Yolo](https://github.com/r4ghu/iOS-CoreML-Yolo)

## YOLOV2

#### YOLO9000: Better, Faster, Stronger
- arxiv: [https://arxiv.org/abs/1612.08242](https://arxiv.org/abs/1612.08242)
- blog: [https://pjreddie.com/darknet/yolov2/](https://pjreddie.com/darknet/yolov2/)[https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
- github(keras): [https://github.com/allanzelener/YAD2K](https://github.com/allanzelener/YAD2K)
- github(pytorch): [https://github.com/longcw/yolo2-pytorch](https://github.com/longcw/yolo2-pytorch)
- github(tensorflow): [https://github.com/hizhangp/yolo_tensorflow](https://github.com/hizhangp/yolo_tensorflow)
- github(windows): [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- github(caffe): [https://github.com/choasUp/caffe-yolo9000](https://github.com/choasUp/caffe-yolo9000)
- github(tensorflow): [https://github.com/KOD-Chen/YOLOv2-Tensorflow](https://github.com/KOD-Chen/YOLOv2-Tensorflow)
- github(tensorflow): [https://github.com/WojciechMormul/yolo2](https://github.com/WojciechMormul/yolo2)

#### darknet_scripts
- intro: Auxilary scripts to work with (YOLO) darknet deep learning famework. AKA -> How to generate YOLO anchors?
- github: [https://github.com/Jumabek/darknet_scripts](https://github.com/Jumabek/darknet_scripts)

#### Yolo_mark: GUI for marking bounded boxes of objects in images for training Yolo v2
- github: [https://github.com/AlexeyAB/Yolo_mark](https://github.com/AlexeyAB/Yolo_mark)

#### LightNet: Bringing pjreddie's DarkNet out of the shadows:
- github: [https://github.com//explosion/lightnet](https://github.com//explosion/lightnet)

#### Loss Rank Mining: A General Hard Example Mining Method for Real-time Detectors:
- intro: LRM is the first hard example mining strategy which could fit YOLOv2 perfectly and make it better applied in series of real scenarios where both real-time rates and accurate detection are strongly demanded.
- arxiv: [https://arxiv.org/abs/1804.04606](https://arxiv.org/abs/1804.04606)

#### Object detection at 200 Frames Per Second
- intro: faster than Tiny-Yolo-v2
- arxiv: [https://arxiv.org/abs/1805.06361](https://arxiv.org/abs/1805.06361)

#### OmniDetector: With Neural Networks to Bounding Boxes
- intro: a person detector on n fish-eye images of indoor scenes（NIPS 2018）
- arxiv: [https://arxiv.org/abs/1805.08503](https://arxiv.org/abs/1805.08503)
- dataset: [https://gitlab.com/auxilia/omnidetector](https://gitlab.com/auxilia/omnidetector)


## YOLOV3

#### YOLOv3: An Incremental Improvement
- arxiv: [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)
- paper: [https://pjreddie.com/media/files/papers/YOLOv3.pdf](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- official_blof: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
- github(official): [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)
- github(tensorflow): [https://github.com/mystic123/tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3)
- github(keras): [https://github.com/experiencor/keras-yolo3](https://github.com/experiencor/keras-yolo3)
- github(keras): [https://github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
- github(pytorch): [https://github.com/ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)
- github(pytorch from scratch): [https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch)
- github(pytorch): [https://github.com/eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

## SSD

#### SSD: Single Shot MultiBox Detector
- intro: ECCV 2016 Oral
- arxiv: [https://arxiv.org/abs/1512.02325](https://arxiv.org/abs/1512.02325)
- github(official): [https://github.com/weiliu89/caffe/tree/ssd](https://github.com/weiliu89/caffe/tree/ssd)
- github(mxnet): [https://github.com/zhreshold/mxnet-ssd](https://github.com/zhreshold/mxnet-ssd)
- github(keras): [https://github.com/rykov8/ssd_keras](https://github.com/rykov8/ssd_keras)
- github(tensorflow): [https://github.com/balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)
- github(mobilnet-ssd/caffe): [https://github.com/chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)
- github(pytorch): [https://github.com/amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)

#### What's the diffience in performance between this new code you pushed and the previous code? #327
- [https://github.com/weiliu89/caffe/issues/327](https://github.com/weiliu89/caffe/issues/327)


## Mask R-CNN
- arxiv: [https://arxiv.org/abs/1703.06870]
- github(keras ans tensorflow): [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- github(pytorch): [https://github.com/wannabeOG/Mask-RCNN8](https://github.com/wannabeOG/Mask-RCNN)
- github(mxnet): [https://github.com/TuSimple/mx-maskrcnn](https://github.com/TuSimple/mx-maskrcnn)

## Mask Scoring R-CNN
- arxiv: [https://arxiv.org/abs/1903.00241](https://arxiv.org/abs/1903.00241)

## Feature Pyramid Networks for Object Detection(FPN)
- paper: [https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)










