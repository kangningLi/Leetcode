## YOLOV1 

**YOLOv1** is a single-stage object detection model. Object detection is framed as a regression problem to spatially separated bounding boxes and associated class probabilities. **A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation**. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance. 

The network uses features from the entire image to predict each bounding box. It also predicts all bounding boxes across all classes for an image simultaneously. This means the network reasons globally about the full image and all the objects in the image.

整体来看，Yolo算法采用一个单独的CNN模型实现end-to-end的目标检测，整个系统如图5所示：首先将输入图片resize到448x448，然后送入CNN网络，最后处理网络预测结果得到检测的目标。相比R-CNN算法，其是一个统一的框架，其速度更快，而且Yolo的训练过程也是end-to-end的。

![img](https://pic4.zhimg.com/80/v2-d37bcff4e377a514aabfb0e371ccdf7b_720w.jpg)图5 Yolo检测系统

input： 448*448

Our system divides the input image into an S × S grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.（Yolo的CNN网络将输入的图片分割成 ![[公式]](https://www.zhihu.com/equation?tex=S%5Ctimes+S) 网格，然后每个单元格负责去检测那些中心点落在该格子内的目标，可以看到狗这个目标的中心落在左下角一个单元格内，那么该单元格负责预测这个狗。）

Each grid cell predicts B bounding boxes and confidence scores for those boxes （每个单元格会预测 ![[公式]](https://www.zhihu.com/equation?tex=B) 个边界框（bounding box）以及边界框的置信度（confidence score）. 

These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts. Formally we define confidence as Pr(Object) ∗ IOUtruth. If no pred object exists in that cell, the confidence scores should be zero.所谓置信度其实包含两个方面，一是这个边界框含有目标的可能性大小，二是这个边界框的准确度。前者记为 ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29) ，**当该边界框是背景时（即不包含目标），此时 ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29%3D0) 。而当该边界框包含目标时， ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29%3D1) 。****边界框的准确度可以用预测框与实际框（ground truth）的IOU（intersection over union，交并比）来表征，记为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D) 。因此置信度可以定义为 ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29%2A%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D) 。**很多人可能将Yolo的置信度看成边界框是否含有目标的概率，但是其实它是两个因子的乘积，预测框的准确度也反映在里面。



边界框的大小与位置可以用4个值来表征： ![[公式]](https://www.zhihu.com/equation?tex=%28x%2C+y%2Cw%2Ch%29) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29) 是边界框的中心坐标，而 ![[公式]](https://www.zhihu.com/equation?tex=w) 和 ![[公式]](https://www.zhihu.com/equation?tex=h) 是边界框的宽与高。还有一点要注意，中心坐标的预测值 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29) 是相对于每个单元格左上角坐标点的偏移值，并且单位是相对于单元格大小的，单元格的坐标定义如图6所示。而边界框的 ![[公式]](https://www.zhihu.com/equation?tex=w) 和 ![[公式]](https://www.zhihu.com/equation?tex=h) 预测值是相对于整个图片的宽与高的比例，这样理论上4个元素的大小应该在 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D) 范围。这样，每个边界框的预测值实际上包含5个元素： ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%2Cw%2Ch%2Cc%29) ，其中前4个表征边界框的大小与位置，而最后一个值是置信度。

![img](https://picb.zhimg.com/80/v2-fdfea5fcb4ff3ecc327758878e4ad6e1_720w.jpg)图6 网格划分

还有分类问题，对于每一个单元格其还要给出预测出 ![[公式]](https://www.zhihu.com/equation?tex=C) 个类别概率值，其表征的是由该单元格负责预测的边界框其目标属于各个类别的概率。但是这些概率值其实是在各个边界框置信度下的条件概率，即 ![[公式]](https://www.zhihu.com/equation?tex=Pr%28class_%7Bi%7D%7Cobject%29) 。值得注意的是，不管一个单元格预测多少个边界框，其只预测一组类别概率值，这是Yolo算法的一个缺点，在后来的改进版本中，Yolo9000是把类别概率预测值与边界框是绑定在一起的。同时，我们可以计算出各个边界框类别置信度（class-specific confidence scores）: ![[公式]](https://www.zhihu.com/equation?tex=Pr%28class_%7Bi%7D%7Cobject%29%2APr%28object%29%2A%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D%3DPr%28class_%7Bi%7D%29%2A%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D) 。

边界框类别置信度表征的是该边界框中目标属于各个类别的可能性大小以及边界框匹配目标的好坏。后面会说，一般会根据类别置信度来过滤网络的预测框。

总结一下，每个单元格需要预测 ![[公式]](https://www.zhihu.com/equation?tex=%28B%2A5%2BC%29) 个值。如果将输入图片划分为 ![[公式]](https://www.zhihu.com/equation?tex=S%5Ctimes+S) 网格，那么最终预测值为 ![[公式]](https://www.zhihu.com/equation?tex=S%5Ctimes+S%5Ctimes+%28B%2A5%2BC%29) 大小的张量。整个模型的预测值结构如下图所示。对于PASCAL VOC数据，其共有20个类别，如果使用 ![[公式]](https://www.zhihu.com/equation?tex=S%3D7%2CB%3D2) ，那么最终的预测结果就是 ![[公式]](https://www.zhihu.com/equation?tex=7%5Ctimes+7%5Ctimes+30) 大小的张量。在下面的网络结构中我们会详细讲述每个单元格的预测值的分布位置。

![img](https://pic2.zhimg.com/80/v2-258df167ee37b5594c72562b4ae61d1a_720w.jpg)图7 模型预测值结构

## 网络设计



**Yolo采用卷积网络来提取特征，然后使用全连接层来得到预测值。网络结构参考GooLeNet模型，包含24个卷积层和2个全连接层**，如图8所示。**对于卷积层，与googlenet使用inception不同，主要使用1x1卷积来做channle reduction，然后紧跟3x3卷积。对于卷积层和全连接层，采用Leaky ReLU激活函数： ![[公式]](https://www.zhihu.com/equation?tex=max%28x%2C+0.1x%29) 。但是最后一层却采用线性激活函数。**

![img](https://picb.zhimg.com/80/v2-5d099287b1237fa975b1c19bacdfc07f_720w.jpg)图8 网络结构

Figure 3: The Architecture. Our detection network has 24 convolutional layers followed by 2 fully connected layers. Alternating 1 × 1 convolutional layers reduce the features space from preceding layers. We pretrain the convolutional layers on the ImageNet classification task at half the resolution (224 × 224 input image) and then double the resolution for detection.

#### Loss function：

**核心是sum-squared error**

**Yolo算法将目标检测看成回归问题，所以采用的是均方差损失函数。但是对不同的部分采用了不同的权重值。**

we **increase the loss from bounding box coordinate predictions and decrease the loss from confidence predictions for boxes that don’t contain objects**. We use two parameters, λcoord and λnoobj to accomplish this. We set λcoord = 5 and λnoobj = 0.5

Sum-squared error also equally weights errors **in large boxes and small boxe**s. Our error metric should reflect that small deviations in large boxes matter less than in small boxes. To partially address this we predict the square root of the bounding box width and height instead of the width and height directly.

另外一点时，**由于每个单元格预测多个边界框。但是其对应类别只有一个**。**那么在训练时，如果该单元格内确实存在目标，那么只选择与ground truth的IOU最大的那个边界框来负责预测该目标**，（anchors）而其它边界框认为不存在目标。这样设置的一个结果将会使一个单元格对应的边界框更加专业化，其可以分别适用不同大小，不同高宽比的目标，从而提升模型性能。大家可能会想如果一个单元格内存在多个目标怎么办，其实这时候Yolo算法就只能选择其中一个来训练，这也是Yolo算法的缺点之一。要注意的一点时，对于不存在对应目标的边界框，其误差项就是只有置信度，坐标项误差是没法计算的。而只有当一个单元格内确实存在目标时，才计算分类误差项，否则该项也是无法计算的。

**YOLO predicts multiple bounding boxes per grid cell.** **At training time we only want one bounding box predictor to be responsible for each object**. We assign one predictor to be “responsible” for predicting an object based on which prediction has the highest current IOU with the ground truth. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall.



综上讨论，最终的损失函数计算如下：（multi part loss function）

![img](https://pic4.zhimg.com/80/v2-45795a63cdbaac8c05d875dfb6fcfb5a_720w.jpg)

其中第一项是边界框中心坐标的误差项， ![[公式]](https://www.zhihu.com/equation?tex=1%5E%7Bobj%7D_%7Bij%7D) 指的是第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个单元格存在目标，且该单元格中的第 ![[公式]](https://www.zhihu.com/equation?tex=j) 个边界框负责预测该目标。第二项是边界框的高与宽的误差项。第三项是包含目标的边界框的置信度误差项。第四项是不包含目标的边界框的置信度误差项。而最后一项是包含目标的单元格的分类误差项， ![[公式]](https://www.zhihu.com/equation?tex=1%5E%7Bobj%7D_%7Bi%7D) 指的是第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个单元格存在目标。这里特别说一下置信度的target值 ![[公式]](https://www.zhihu.com/equation?tex=C_i)，如果是不存在目标，此时由于 ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29%3D0)，那么 ![[公式]](https://www.zhihu.com/equation?tex=C_i%3D0) 。如果存在目标， ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29%3D1) ，此时需要确定 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BIOU%7D%5E%7Btruth%7D_%7Bpred%7D) ，当然你希望最好的话，可以将IOU取1，这样 ![[公式]](https://www.zhihu.com/equation?tex=C_i%3D1)，但是在YOLO实现中，使用了一个控制参数rescore（默认为1），当其为1时，IOU不是设置为1，而就是计算truth和pred之间的真实IOU。不过很多复现YOLO的项目还是取 ![[公式]](https://www.zhihu.com/equation?tex=C_i%3D1) ，这个差异应该不会太影响结果吧。



下面就来分析Yolo的预测过程，这里我们不考虑batch，认为只是预测一张输入图片。根据前面的分析，最终的网络输出是 ![[公式]](https://www.zhihu.com/equation?tex=7%5Ctimes+7+%5Ctimes+30) ，但是我们可以将其分割成三个部分：类别概率部分为 ![[公式]](https://www.zhihu.com/equation?tex=%5B7%2C+7%2C+20%5D) ，置信度部分为 ![[公式]](https://www.zhihu.com/equation?tex=%5B7%2C7%2C2%5D) ，而边界框部分为 ![[公式]](https://www.zhihu.com/equation?tex=%5B7%2C7%2C2%2C4%5D) （对于这部分不要忘记根据原始图片计算出其真实值）。然后将前两项相乘（矩阵 ![[公式]](https://www.zhihu.com/equation?tex=%5B7%2C+7%2C+20%5D) 乘以 ![[公式]](https://www.zhihu.com/equation?tex=%5B7%2C7%2C2%5D) 可以各补一个维度来完成 ![[公式]](https://www.zhihu.com/equation?tex=%5B7%2C7%2C1%2C20%5D%5Ctimes+%5B7%2C7%2C2%2C1%5D) ）可以得到类别置信度值为 ![[公式]](https://www.zhihu.com/equation?tex=%5B7%2C+7%2C2%2C20%5D) ，这里总共预测了 ![[公式]](https://www.zhihu.com/equation?tex=7%2A7%2A2%3D98) 个边界框。

#### Limitations of YOLOV1

首先Yolo各个单元格仅仅预测两个边界框，而且属于一个类别。对于小物体，Yolo的表现会不如人意。这方面的改进可以看SSD，其采用多尺度单元格。也可以看Faster R-CNN，其采用了anchor boxes。Yolo对于在物体的宽高比方面泛化率低，就是无法定位不寻常比例的物体。当然Yolo的定位不准确也是很大的问题。

**一个grid cell中是否有object怎么界定？** 
首先要明白grid cell的含义，以文中7*7为例，这个size其实就是对输入图像（假设是224*224）不断提取特征然后sample得到的（缩小了32倍），然后就是把输入图像划分成7*7个grid cell，这样输入图像中的32个像素点就对应一个grid cell。回归正题，那么我们有每个object的标注信息，也就是知道每个object的中心点坐标在输入图像的哪个位置，那么不就相当于知道了每个object的中心点坐标属于哪个grid cell了吗，而只要object的中心点坐标落在哪个grid cell中，这个object就由哪个grid cell负责预测，也就是该grid cell包含这个object。另外由于一个grid cell会预测两个bounding box，实际上只有一个bounding box是用来预测属于该grid cell的object的，因为这两个bounding box到底哪个来预测呢？答案是：和该object的ground truth的IOU值最大的bounding box。



## YOLOV2

**YOLOv2**, or [**YOLO9000**](https://www.youtube.com/watch?v=QsDDXSmGJZA), is a single-stage real-time object detection model. It improves upon [YOLOv1](https://paperswithcode.com/method/yolov1) in several ways, including the **use of Darknet-19 as a backbone, batch normalization, use of a high-resolution classifier, and the use of anchor boxes to predict bounding boxes, and more.**

input：416 * 416

#### 对yolo的改进：

1. backbone

   propose darknet19, 参考了前人的工作经验。类似于VGG，网络使用了较多的3  * *3卷积核，在每一次池化操作后把通道数翻倍。借鉴了Network In Network的思想，网络使用了全局平均池化（Global Average Pooling）做预测，把1* * 1的卷积核置于3*3的卷积核之间，用来压缩特征。使用Batch Normalization稳定模型训练，加速收敛，正则化模型。

   最终得出的基础模型就是Darknet-19，包含19个卷积层、5个最大值池化层（Max Pooling Layers ）。Darknet-19处理一张照片需要55.8亿次运算，Imagenet的Top-1准确率为72.9%，Top-5准确率为91.2%。

2. batch-normalization

   相对于YOLOv1，YOLOv2将`dropout`替换成了效果更好的`batch normalization`，在每个卷积层计算之前利用`batch normalization`进行批归一化：

   ![[公式]](https://www.zhihu.com/equation?tex=%7B%7B%5Cmathord%7B%5Cbuildrel%7B%5Clower3pt%5Chbox%7B%24%5Cscriptscriptstyle%5Cfrown%24%7D%7D++%5Cover+x%7D+%7D%5Ek%7D+%3D+%5Cfrac%7B%7B%7Bx%5Ek%7D+-+E%5Cleft%5B+%7B%7Bx%5Ek%7D%7D+%5Cright%5D%7D%7D%7B%7B%5Csqrt+%7BVar%5Cleft%5B+%7B%7Bx%5Ek%7D%7D+%5Cright%5D%7D+%7D%7D%5C%5C+%7By%5Ek%7D+%3D+%7B%5Cgamma+%5Ek%7D%7B%7B%5Cmathord%7B%5Cbuildrel%7B%5Clower3pt%5Chbox%7B%24%5Cscriptscriptstyle%5Cfrown%24%7D%7D++%5Cover+x%7D+%7D%5Ek%7D+%2B+%7B%5Cbeta+%5Ek%7D)

3. high-resolution classifier

   都要先把分类器（classiﬁer）放在ImageNet上进行预训练。从Alexnet开始，大多数的分类器都运行在小于256*256的图片上。而现在YOLO从224*224增加到了448*448，这就意味着网络需要适应新的输入分辨率。
   为了适应新的分辨率，YOLO v2的分类网络以448*448的分辨率先在ImageNet上进行Fine Tune，Fine Tune10个epochs，让网络有时间调整他的滤波器（filters），好让其能更好的运行在新分辨率上，还需要调优用于检测的Resulting Network。最终通过使用高分辨率，mAP提升了4%。

4. anchor box(用 k means选) anchor box 替换全联接层

   YOLO一代包含有全连接层，从而能直接预测Bounding Boxes的坐标值。 Faster R-CNN的方法只用卷积层与Region Proposal Network来预测Anchor Box的偏移值与置信度，而不是直接预测坐标值。作者发现通过预测偏移量而不是坐标值能够简化问题，让神经网络学习起来更容易。
   所以最终YOLO去掉了全连接层，使用Anchor Boxes来预测 Bounding Boxes。作者去掉了网络中一个Pooling层，这让卷积层的输出能有更高的分辨率。收缩网络让其运行在416*416而不是448*448。由于图片中的物体都倾向于出现在图片的中心位置，特别是那种比较大的物体，所以有一个单独位于物体中心的位置用于预测这些物体。YOLO的卷积层采用32这个值来下采样图片，所以通过选择416*416用作输入尺寸最终能输出一个13*13的Feature Map。 使用Anchor Box会让精确度稍微下降，但用了它能让YOLO能预测出大于一千个框，同时recall达到88%，mAP达到69.2%。

   

   

5. direct location prediction:（绝对位置预测）

RPN中预测坐标就是预测tx，ty，对应中心点（x，y）计算如下：

<img src="https://pic4.zhimg.com/80/v2-9d577424c78f0edf11572ca31256cbb8_720w.jpg" alt="img" style="zoom:30%;" />



可见预测tx=1就会把box向右移动Anchor Box的宽度，预测tx=-1就会把Box向左移动相同的距离。这个公式没有任何限制，无论在什么位置进行预测，任何Anchor Boxes可以在图像中任意一点。模型随机初始化之后将需要很长一段时间才能稳定预测敏感的物体偏移。因此作者没有采用这种方法，而是预测相对于Grid Cell的坐标位置，同时把Ground Truth限制在0到1之间（利用Logistic激活函数约束网络的预测值来达到此限制）。

<img src="http://lanbing510.info/public/img/posts/yolov2/2.png" alt="img" style="zoom:55%;" />

**The network predicts 5 bounding boxes at each cell in the output feature map**. The network predicts 5 coordinates for each bounding box, tx, ty, tw, th, and to. If the cell is offset from the top left corner of the image by (cx,cy) and the bounding box prior has width and height pw, ph, then the predictions correspond to:

用Anchor Box的方法，会让model变得不稳定，尤其是在最开始的几次迭代的时候。大多数不稳定因素产生自预测Box的（x,y）位置的时候。按照之前YOLO的方法，网络不会预测偏移量，而是根据YOLO中的网格单元的位置来预测坐标，这就让Ground Truth的值介于0到1之间。而为了让网络的结果能落在这一范围内，网络使用一个 Logistic Activation来对于网络预测结果进行限制，让结果介于0到1之间。 网络在每一个网格单元中预测出5个Bounding Boxes，每个Bounding Boxes有五个坐标值tx，ty，tw，th，t0，他们的关系见下图（Figure3）。假设一个网格单元对于图片左上角的偏移量是cx，cy，Bounding Boxes Prior的宽度和高度是pw，ph，那么预测的结果见下图右面的公式：如果这个Cell距离图像左上角的边距为(cx,cy)以及该Cell对应的Box维度（Bounding Box Prior）的长和宽分别为(pw,ph)，bx,by,bw,bh的预测值见下图，Pr(object)∗IOU(b,object)=σ(t0)。

![这里写图片描述](https://img-blog.csdn.net/20161228203420288?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaHlzdGVyaWMzMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



神经网络预测 ![[公式]](https://www.zhihu.com/equation?tex=t_x+%2Ct_y) ，而![[公式]](https://www.zhihu.com/equation?tex=t_x+%2Ct_y)又需要与先验框的宽高相乘才能得到相较于 ![[公式]](https://www.zhihu.com/equation?tex=x_a%2C+y_a) 的位置偏移值，在v2中，位置预测公式如下见上图：

6. dimension clustering（Anchor Box的宽高由聚类产生），k为anchor的个数，以ground truth的box为中心

   Instead of choosing priors by hand, we run **k-means clustering** on the training set bounding boxes to automatically find good priors. If we use standard k-means with Euclidean distance larger boxes generate more error than smaller boxes. However, what we really want are priors that lead to good IOU scores, which is independent of the size of the box. Thus for our distance metric we use:

   **d(box, centroid) = 1 − IOU(box, centroid)**

   Anchor Box的宽高不经过人为获得，而是将训练数据集中的矩形框全部拿出来，用kmeans聚类得到先验框的宽和高。例如使用5个Anchor Box，那么kmeans聚类的类别中心个数设置为5。加入了聚类操作之后，引入Anchor Box之后，mAP上升。

   需要强调的是，聚类必须要定义聚类点（矩形框 ![[公式]](https://www.zhihu.com/equation?tex=%28w%2Ch%29)）之间的距离函数，文中使用如下函数：

   ![img](https://pic4.zhimg.com/80/v2-418f012af6cc4f94db0486cdca8968aa_720w.png)

   使用（1-IOU）数值作为两个矩形框的的距离函数，这里的运用也是非常的巧妙。

7. Fine-grained features（expand feature map）

   This modified YOLO predicts detections on a 13 × 13 feature map

   turns the 26 × 26 × 512 feature map into a 13 × 13 × 2048 feature map, which can be concatenated with the original features.

   ssd 和 faster rcnn中都用了multi-scale feature map, 这里yolo的做法不同，加了一个passthrough layer， simply adding a passthrough layer that brings features from an earlier layer at 26 × 26 resolution.

   Passthrough Laye把高低分辨率的特征图做连结，叠加相邻特征到不同通道（而非空间位置）， 类似于Resnet中的Identity Mappings。这个方法把26* 26*512的特征图叠加成13* 13*2048的特征图（concatenate），与原生的深层特征图相连接。

8. 由于网络只用到了卷积层和池化层，就可以进行动态调整（检测任意大小图片, Instead of fixing the input image size we change the net-work every few iterations. Every 10 batches our network randomly chooses a new image dimension size.

   Since our model downsamples by a factor of 32, we pull from the following multiples of 32: {320, 352, ..., 608}. Thus the smallest option is 320 × 320 and the largest is 608 × 608. We resize the network to that dimension and continue training.

9. 作者提出了一种在分类数据集和检测数据集上联合训练(joint training on detection and classification data)的机制。使用检测数据集的图片去学习检测相关的信息，例如Bounding Box 坐标预测，是否包含物体以及属于各个物体的概率。使用仅有类别标签的分类数据集图片去扩展可以检测的种类。

   训练过程中把监测数据和分类数据混合在一起。当网络遇到一张属于检测数据集的图片就基于YOLOv2的全部损失函数（包含分类部分和检测部分）做反向传播。当网络遇到一张属于分类数据集的图片就仅基于分类部分的损失函数做反向传播。

10. **softmax**
    YOLOv2中对于每个类别的概率输出进行了softmax归一化。

11. ## 优缺点

    YOLOv2相对来说在每个网格内预测了更多的目标框，并且每个目标框可以不用为同一类，而每个目标都有着属于自己的分类概率，这些使得预测结果更加丰富。另外，由于`anchor box`的加入，使得YOLOv2的定位精度更加准确。不过，其对于YOLOv1的许多问题依旧没有解决，当然那些也是很多目标检测算法的通病。那么随着`anchor box`的加入所带来的新问题是：

    - `anchor box`的个数以及参数都属于超参数，因此会影响训练结果；
    - 由于`anchor box`在每个网格内都需要计算一次损失函数，然而每个正确预测的目标框才能匹配一个比较好的先验anchor，也就是说，对于YOLOv2中的5种`anchor box`，相当于强行引入了4倍多的负样本，在本来就样本不均衡的情况下，加重了不均衡程度，从而使得训练难度增大；
    - 由于IOU和NMS的存在，会出现下面的情况：

    ![img](https://pic4.zhimg.com/80/v2-f1bf340437c4951931df052e5cc9734f_720w.jpg)


    我们可以看到，当两个人很靠近或重叠时，检测框变成了中间的矩形框，其原因在于对于两个候选框（红，绿），其中红色框可能更加容易受到目标1的影响，而绿色框会同时收到目标1和目标2的影响，从而导致最终定位在中间。然后由于NMS存在，其他的相邻的框则会被剔除。要想避免这种情况，就应该在损失函数中加入相关的判定。



## YOLOV3

1. darknet53

   ![img](https://pic3.zhimg.com/80/v2-949e446f5a1ab632b9168885a536504c_720w.jpg)

可以看到，新增了`Residual`模块，不同于原本的Resnet中的残差模块： successive 3 × 3 and 1 × 1 convolutional layers



<img src="https://pic1.zhimg.com/80/v2-8e5925a963e9234711e3c4ac61deaf79_720w.jpg" alt="img"  />

2. no softmax

3. YOLOv3 predicts an objectness score for each bounding box using logistic regression.

4. #### 多尺度输出Predictions Across Scales

   YOLOv3增加了top down 的多级预测，解决了yolo颗粒度粗，对小目标无力的问题。

   

   ![img](https://pic3.zhimg.com/80/v2-9fd12a06c1f2e8d98284ca28a3a95300_720w.jpg)

   

   可以看到，不仅在不同的感受野范围输出了三种尺度的预测结果，每种预测结果中每个网格包含3个目标框，一共是9个目标框。而且，相邻尺度的网络还存在着级联：

   YOLOv3 predicts boxes at 3 different scales， Our system extracts features from those scales using a similar concept to feature pyramid networks [8]. From our base feature extractor we add several convolutional layers. The last of these predicts a 3-d tensor encoding bounding box, objectness, and class predictions. In our experiments with COCO [10] we predict 3 boxes at each scale so the tensor is N × N × [3 ∗ (4 + 1 + 80)] for the 4 bounding box offsets, 1 objectness prediction, and 80 class predictions

   Next we take the feature map from 2 layers previous and upsample it by 2×. We also take a feature map from earlier in the network and merge it with our upsampled features using concatenation. This method allows us to get more meaningful semantic information from the upsampled features and finer-grained information from the earlier feature map. We then add a few more convolutional layers to process this combined feature map, and eventually predict a similar tensor, although now twice the size.

   We perform the same design one more time to predict boxes for the final scale. Thus our predictions for the 3rd scale benefit from all the prior computation as well as finegrained features from early on in the network.

   可以看到，不仅在不同的感受野范围输出了三种尺度的预测结果，每种预测结果中每个网格包含3个目标框，一共是9个目标框。而且，相邻尺度的网络还存在着级联：

   

   ![img](https://pic3.zhimg.com/80/v2-943c2e67c28302127a52a0df3e38e6b3_720w.jpg)

   

   **DBL**: conv+BN+Leaky relu。

   **resn**：n代表数字，有res1，res2, … ,res8等等，表示这个res_block里含有多少个res_unit。这是yolo_v3的大组件，yolo_v3开始借鉴了ResNet的残差结构，使用这种结构可以让网络结构更深(从v2的darknet-19上升到v3的darknet-53，前者没有残差结构)。对于res_block的解释，可以在图1的右下角直观看到，其基本组件也是DBL。

   **concat**：张量拼接。将darknet中间层和后面的某一层的上采样进行拼接。拼接的操作和残差层add的操作是不一样的，拼接会扩充张量的维度，而add只是直接相加不会导致张量维度的改变。

   **upsample**：终于把原来的`reorg`改成了`upsample`，这里的upsample很暴力，很像克罗内克积，即：

   ![img](https://pic3.zhimg.com/80/v2-10e78c380ac46671a0b03394fc70c51e_720w.jpg)

   可以看到每个输出的深度都是255，即3x(80+5)。这种多尺度预测的方式应该是参考的FPN算法。

5. 对anchor box的改进

YOLOv2中是直接预测了目标框相对网格点左上角的偏移，以及`anchor box`的修正量，而在YOLOv3中同样是利用K-means聚类得到了9组`anchor box`，只不过YOLOv2中用的是相对比例，而YOLOv3中用的是绝对大小。那么鉴于我们之前提到的`anchor box`带来的样本不平衡问题，以及绝对大小可能会出现超出图像边界的情况，作者加入了新的判断条件，即对于每个目标预测结果只选择与groundtruth的IOU最大/超过0.5的`anchor`，不考虑其他的`anchor`，从而大大减少了样本不均衡情况。







