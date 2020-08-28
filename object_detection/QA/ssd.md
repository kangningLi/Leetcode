 SSD only needs an input image and ground truth boxes for each object during training.

feature map is in different scale 

in different scaled feature map, prior box or anchor box are pre-difined with different scale and ratio

For each default box, we predict both the shape offsets and the confidences for all object categories ((c1 , c2 , · · · , cp )). At training time, we first match these default boxes to the ground truth boxes. **The model loss is a weighted sum between localization loss (e.g. Smooth L1 [6]) and confidence loss (e.g. Softmax).**

**Model:**

input: 300 * 300 * 3

预测的类别数为自己的类别 + 1， 1 为background， 即没有object

vgg16 + extra feature extraction layers

The SSD approach is based on a feed-forward convolutional network that produces **a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes,** followed by a **non-maximum suppression** step to produce the final detections. (**in every feature map layer they do prediction and regression**)

**Multi-scale feature maps for detection（多尺度特征图用于目标检测）**: after truncated vgg16 backbone, add convolutional feature layers. These layers decrease in size progressively and allow predictions of detections at multiple scales. The convolutional model for predicting detections is different for each layer. 

**Convolutional predictors for detection（用卷积进行检测）:** **Each added feature layer can produce a fixed set of predictions using a set of convolutional filters.** 

与yolo最后采用全联接层不同，ssd直接采用卷积对不同的特征图来提取预测结果。

我们先来看论文上的网络结构图：

![img](https://img-blog.csdn.net/20180401104056385)

​    网络结构比较简单，就是在VGG的基础上改得，前面和VGG一样，但是SSD把VGG的全连接层换成了几个卷积层，把droupout层去除了

For a feature layer of size m × n with p channels, the basic element for predicting parameters of a potential detection is a 3 × 3 × p *small kernel* that produces either a score for a category, or a shape offset relative to the default box coordinates. At each of the m × n locations where the kernel is applied, it produces an output value.（对于形状为 m × n ×  p 的特征图，采用3 × 3 × p 这样小的卷积核得到检测值）

**Default boxes and aspect ratios（prior box）**:We associate a set of default bounding boxes with each feature map cell, for multiple feature maps at the top of the network(每个feature map的每个cell都有一组anchors).  At each feature map cell, we predict the offsets relative to the default box shapes in the cell, as well as the per-class scores that indicate the presence of a class instance in each of those boxes. Specifically, for each box out of k at a given location, we compute c class scores and the 4 offsets relative to the original default box shape. 



The fundamental improvement in speed comes from eliminating bounding box proposals and the subsequent pixel or feature resampling stage. Improvements over competing single-stage methods include using a small convolutional filter to predict object categories and offsets in bounding box locations, using separate predictors (filters) for different aspect ratio detections, and applying these filters to multiple feature maps from the later stages of a network in order to perform detection at multiple scales.

与yolo对比：

1. ssd提取不同尺度的特征图来做检测，大尺度特征图（比较靠前）的特征图来检测小物体，而小尺度特征图来检测大物体
2. ssd 用了不同尺度长宽比的先验框，prior box
3. 与yolo最后采用全联接层不同，ssd直接采用卷积对不同的特征图来提取预测结果。
4. 检测值与yolo也不相同，对于每个单元每个prior box，都输出一套独立的检测值，对应一个bounding box，ssd中background也是一个类别，输出的第一部分是对各个类别的confidence， confidence最高的那个就是预测的类别，输出的第二部分是bounding box的location，包括cx，cy，w，h（shape offset releative to deault box coordinate）。

At each feature map cell, we predict the offsets relative to the default box shapes in the cell, as well as the per-class scores that indicate the presence of a class instance in each of those boxes. Specifically, for each box out of k at a given location, we compute c class scores and the 4 offsets relative to the original default box shape. This results in a total of (c + 4)k filters that are applied around each location in the feature map, yielding (c + 4)kmn outputs for a m × n feature map. （对于一个m × n大小的feature map，共有mn个单元，每个单元pre-define k个先验框，那么每个单元有（c + 4)k个预测值，然后对m × n大小的特征图，需要c + 4)kmn个预测值）

训练过程：

1.先验框匹配：

在训练过程中，首先要确定训练图片中的ground truth与哪个先验框来进行匹配，与之匹配的先验框所对应的边界框将负责预测它，在yolo中，ground truth的中心落在哪个单元格，该单元格中与其iou最大的边界框负责预测它。ssd中不一样，ssd的先验框与groundtruth的匹配原则有亮点，首先，对于图片中的每一个ground truth，找到与其iou最大的先验框，该先验框与他匹配。这样，可以保证每个ground truth一定与某个先验框匹配。通常称与先验框匹配的样本为正样本，反之，如果一个先验框没有与任何ground truth匹配，那么只能与背景匹配，为负样本。ground truth很少，这样负样本很多，回导致正负样本极度不平衡，所以需要第二个原则。第二个原则是，对于剩余的未匹配的先验框，如果某个ground truth的iou大于某个阈值，一般为0.5，那么该先验框也与这个ground truth匹配，这意味着某个groundtruth可能与多个先验框匹配。这是可以的，但是反过来不可以。



虽然一个ground truth可以与多个先验框匹配，但是依旧ground truth太少了。所以负样本会很多，导致正负样本极度不平衡，ssd使用了hard negative mining， 对负样本进行采样，抽样时highest confidence loss for each default box and pick the top ones so that the ratio between the negatives and positives is at most 3:1。（采样top k）



2. data augmentation：

   水平翻转（horizontal flip）， 随机剪裁加颜色扭曲（random crop & color distortion）， 随机采集块域（randomly sample a patch获得小目标训练样本）

预测过程：

对每个预测框，首先根据类别置信度确定其类别，并过滤掉属于背景的预测框，然后根据置信度阈值（如0.5）过滤掉阈值较低的预测框。

然后解码，解码后要根据置信度进行降序排序，然后仅保留top-k个预测框，然后nms。剩余的就是检测结果。