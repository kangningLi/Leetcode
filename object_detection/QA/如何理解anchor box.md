# anchor box

anchor相关的问题：R-CNN，SSD，YOLO中的anchor

预定义边框就是一组预设的边框，在训练时，以真实的边框位置相对于预设边框的偏移来构建
训练样本。 这就相当于，**预设边框先大致在可能的位置“框“出来目标，然后再在这些预设边框的基础上进行调整**。



一个**Anchor Box**可以由:边框的纵横比和边框的面积（尺度)来定义，相当于一系列预设边框的生成规则，根据Anchor Box，可以在图像的任意位置，生成一系列的边框。由于Anchor box 通常是以CNN提取到的Feature Map 的点为中心位置，生成边框，所以一个Anchor box不需要指定中心位置。



总结来说就是：在一幅图像中，要检测的目标可能出现在图像的任意位置，并且目标可能是任意的大小和任意形状。

- 使用CNN提取的Feature Map的点，来定位目标的位置。
- 使用Anchor box的**Scale**来表示目标的大小
- 使用Anchor box的**Aspect Ratio**来表示目标的形状



### 常用的Anchor Box定义

- Faster R-CNN 定义三组纵横比`ratio = [0.5,1,2]`和三种尺度`scale = [8,16,32]`，可以组合处9种不同的形状和大小的边框。
- YOLO V2 V3 则不是使用预设的纵横比和尺度的组合，而是使用`k-means`聚类的方法，从训练集中学习得到不同的Anchor
- SSD 固定设置了5种不同的纵横比`ratio=[1,2,3,1/2,1/3]`,由于使用了多尺度的特征，对于每种尺度只有一个固定的`scale`

### Anchor 的意义

Anchor Box的生成是以CNN网络最后生成的Feature Map上的点为中心的（映射回原图的坐标），以Faster R-CNN为例，使用VGG网络对对输入的图像下采样了**16**倍，也就是Feature Map上的一个点对应于输入图像上的一个16×16的正方形区域（感受野）。根据预定义的Anchor,Feature Map上的一点为中心 就可以在原图上生成9种不同形状不同大小的边框，如下图：

![anchor](https://github.com/kangningLi/paperList/blob/master/object_detection/image/anchor.png)

从上图也可以看出为什么需要Anchor。根据CNN的感受野，一个Feature Map上的点对应于原图的16×16的正方形区域，仅仅利用该区域的边框进行目标定位，其精度无疑会很差，甚至根本“框”不到目标。 而加入了Anchor后，一个Feature Map上的点可以生成9中不同形状不同大小的框，这样“框”住目标的概率就会很大，就大大的提高了检查的召回率；再通过后续的网络对这些边框进行调整，其精度也能大大的提高。

## YOLO 的Anchor Box

YOLO v2,v3的Anchor Box 的大小和形状是通过对训练数据的聚类得到的。 作者发现如果采用标准的k-means（即用欧式距离来衡量差异），在box的尺寸比较大的时候其误差也更大，而我们希望的是误差和box的尺寸没有太大关系。这里的意思是不能直接使用𝑥,𝑦,𝑤,ℎx,y,w,h这样的四维数据来聚类，因为框的大小不一样，这样大的定位框的误差可能更大，小的定位框误差会小，这样不均衡，很难判断聚类效果的好坏。
所以通过IOU定义了如下的距离函数，使得误差和box的大小无关：

𝑑(𝑏𝑜𝑥,𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑)=1−𝐼𝑂𝑈(𝑏𝑜𝑥,𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑)d(box,centroid)=1−IOU(box,centroid)

官方的 V2，V3的Anchor

```
anchors = 0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828 // yolo v2

anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326 // yolo v3
```

需要注意的是 anchor计算的尺度问题。 yolo v2的是相对最后一个Feature Map (13×13)(13×13)来的，yolo v3则是相对于原输入图像的416×416的。 这也是在计算目标检测中，边框计算中需要注意的问题，就是计算的边框究竟在相对于那个尺度得出的。

## SSD的 default Prior box

SSD的Prior Box的三要素：

- Feature Map
  为了能够更好的检测小目标，SSD使用不同层的Feature Map（不同尺度）。具体就是：38×38,19 *19,10×10,5×5,3×3,1×1

- **Scale**

  - 假设一个prior box的scale为𝑠，这就表示该prior box的面积是以𝑠为边长的正方形的面积。
  - scale的值是以相对于原始图像的边长比例来设定的。 例如$ s= 0.1，则表示实际，则表示实际s = 0.1 \cdot W_src$。
  - SSD中使用了多个不同尺度的Feature Map。针对不同的Feature Map，设定不同的尺度。除了第一个Feature Map(38×3838×38)的𝑠𝑐𝑎𝑙𝑒=0.1scale=0.1，剩余各层的按照Feature Map的尺度从大大小排列，其scale按照下面的公式增大

  𝑠𝑐𝑎𝑙𝑒=𝑠𝑚𝑖𝑛+𝑠𝑚𝑎𝑥−𝑠𝑚𝑖𝑛𝑚−1⋅(𝑘−1),scale=smin+smax−sminm−1⋅(k−1),

  其中，𝑚=5,𝑘=1,2,3,4,5,𝑠𝑚𝑖𝑛=0.2,𝑠𝑚𝑎𝑥=0.9m=5,k=1,2,3,4,5,smin=0.2,smax=0.9
  可以看到，尺度大的Feature Map其scale较小，利于小目标的检测。

- **Aspect Ratio**
  每个尺度的Feature Map的prior box都有(1:1,2:1,1:2)(1:1,2:1,1:2)这三种aspect ratio。其中，19×19,10×10,5×519×19,10×10,5×5这三个Feature Map则有额外的两个aspect ratio (1:3,3:1)(1:3,3:1)。

- **附加的prior box**
  针对所有的Feature Map 都有一个附加的 prior box，其aspect ratio为1:11:1，其尺度为当前Feature Map 和下一个Feature Map的几何平均值，也就是(√𝑠𝑐𝑎𝑙𝑒𝑘⋅𝑠𝑐𝑎𝑙𝑒𝑘+1)(scalek⋅scalek+1)

SSD的Prior box的总结如下：
![img](https://img2018.cnblogs.com/blog/439761/201912/439761-20191209174221680-1022296075.png)

## Anchor机制又是什么？

以每个anchor为中心点，人为设置不同的尺度(scale)和长宽比(aspect ratio)，即可得到基于anchor的多个anchor box，用以框定图像中的目标，这就是所谓的anchor 机制。

## Anchor机制的优缺点

知道了anchor和anchor机制是什么，那接下来就说说它在检测任务中起到的优缺点。这里我们总结了以下几个方面：

\1. 优点：

（1）使用anchor机制产生密集的anchor box，使得网络可直接在此基础上进行目标分类及边界框坐标回归；

（2）密集的anchor box可有效提高网络目标召回能力，对于小目标检测来说提升非常明显。

\2. 缺点：

（1）anchor机制中，需要设定的超参：尺度(scale)和长宽比( aspect ratio)是比较难设计的。这需要较强的先验知识。

（2）冗余框非常之多：一张图像内的目标毕竟是有限的，基于每个anchor设定大量anchor box会产生大量的easy-sample，即完全不包含目标的背景框。这会造成正负样本严重不平衡问题，也是one-stage算法难以赶超two-stage算法的原因之一。

（3）网络实质上是看不见anchor box的，在anchor box的基础上进行边界回归更像是一种在范围比较小时候的强行记忆。

（4）基于anchor box进行目标类别分类时，IOU阈值超参设置也是一个问题，0.5？0.7？有同学可能也想到了CVPR2018的论文Cascade R-CNN，专门来讨论这个问题。感兴趣的同学可以移步：[Naiyan Wang：CVPR18 Detection文章选介（上）](https://zhuanlan.zhihu.com/p/35882192)

问题出现了，接下来我们将介绍针对Anchor机制的不足，涌现出的大佬们的解决方案，并且尝试去解读他们。







