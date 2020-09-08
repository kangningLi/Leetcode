# **Feature Pyramid Network**

A **Feature Pyramid Network**, or **FPN**, is a feature extractor that takes a single-scale image of an arbitrary size as input, and outputs proportionally sized feature maps at multiple levels, in a fully convolutional fashion. This process is independent of the backbone convolutional architectures. It therefore acts as a generic solution for building feature pyramids inside deep convolutional networks to be used in tasks like object detection.

The construction of the pyramid involves a bottom-up pathway and a top-down pathway.

The bottom-up pathway is the feedforward computation of the backbone ConvNet, which computes a feature hierarchy consisting of feature maps at several scales with a scaling step of 2. For the feature pyramid, one pyramid level is defined for each stage. The output of the last layer of each stage is used as a reference set of feature maps. For [ResNets](https://paperswithcode.com/method/resnet) we use the feature activations output by each stage’s last residual block.

The top-down pathway hallucinates higher resolution features by upsampling spatially coarser, but semantically stronger, feature maps from higher pyramid levels. These features are then enhanced with features from the bottom-up pathway via lateral connections. Each lateral connection merges feature maps of the same spatial size from the bottom-up pathway and the top-down pathway. The bottom-up feature map is of lower-level semantics, but its activations are more accurately localized as it was subsampled fewer times.

## **为什么需要构造特征金字塔？**

**![img](https://img-blog.csdn.net/20180309183706872)**

图3 特征金字塔

前面已经提到了高斯金字塔，由于它可以在一定程度上面提高算法的性能，因此很多经典的算法中都包含它。但是这些都是在传统的算法中使用，当然也可以将这种方法直应用在深度神经网络上面，但是由于它需要大量的运算和大量的内存。但是**我们的特征金字塔可以在速度和准确率之间进行权衡，可以通过它获得更加鲁棒的语义信息**，这是其中的一个原因。

如上图所示，我们可以看到我们的**图像中存在不同尺寸的目标，而不同的目标具有不同的特征，利用浅层的特征就可以将简单的目标的区分开来；利用深层的特征可以将复杂的目标区分开来；**这样我们就需要这样的一个特征金字塔来完成这件事。图中我们在第1层（请看绿色标注）输出较大目标的实例分割结果，在第2层输出次大目标的实例检测结果，在第3层输出较小目标的实例分割结果。检测也是一样，我们会在第1层输出简单的目标，第2层输出较复杂的目标，第3层输出复杂的目标。

下图FIg1展示了4种利用特征的形式：
（a）图像金字塔，即将图像做成不同的scale，然后不同scale的图像生成对应的不同scale的特征。这种方法的缺点在于增加了时间成本。有些算法会在测试时候采用图像金字塔。
（b）像SPP net，Fast RCNN，Faster RCNN是采用这种方式，即仅采用网络最后一层的特征。
（c）像SSD（Single Shot Detector）采用这种多尺度特征融合的方式，没有上采样过程，即从网络不同层抽取不同尺度的特征做预测，这种方式不会增加额外的计算量。作者认为SSD算法中没有用到足够低层的特征（在SSD中，最低层的特征是VGG网络的conv4_3），而在作者看来足够低层的特征对于检测小物体是很有帮助的。
（d）本文作者是采用这种方式，顶层特征通过上采样和低层特征做融合，而且每层都是独立预测的。

![img](https://img-blog.csdn.net/20180309175744421)



识别不同大小的物体是计算机视觉中的一个基本挑战，我们常用的解决方案是构造多尺度金字塔。如上图a所示，这是一个特征图像金字塔，整个过程是先对原始图像构造图像金字塔，然后在图像金字塔的每一层提出不同的特征，然后进行相应的预测（BB的位置）。这种方法的缺点是计算量大，需要大量的内存；优点是可以获得较好的检测精度。它通常会成为整个算法的性能瓶颈，由于这些原因，当前很少使用这种算法。如上图b所示，这是一种改进的思路，学者们发现我们可以利用卷积网络本身的特性，即对原始图像进行卷积和池化操作，通过这种操作我们可以获得不同尺寸的feature map，这样其实就类似于在图像的特征空间中构造金字塔。实验表明，**浅层的网络更关注于细节信息，高层的网络更关注于语义信息**，**而高层的语义信息能够帮助我们准确的检测出目标，因此我们可以利用最后一个卷积层上的feature map来进行预测**。**这种方法存在于大多数深度网络中，比如VGG、ResNet、Inception，它们都是利用深度网络的最后一层特征来进行分类。这种方法的优点是速度快、需要内存少。它的缺点是我们仅仅关注深层网络中最后一层的特征，却忽略了其它层的特征，但是细节信息可以在一定程度上提升检测的精度。****因此有了图c所示的架构，它的设计思想就是同时利用低层特征和高层特征，分别在不同的层同时进行预测，这是因为我的一幅图像中可能具有多个不同大小的目标，区分不同的目标可能需要不同的特征，对于简单的目标我们仅仅需要浅层的特征就可以检测到它，对于复杂的目标我们就需要利用复杂的特征来检测它。整个过程就是首先在原始图像上面进行深度卷积，然后分别在不同的特征层上面进行预测。它的优点是在不同的层上面输出对应的目标，不需要经过所有的层才输出对应的目标（即对于有些目标来说，不需要进行多余的前向操作），这样可以在一定程度上对网络进行加速操作，同时可以提高算法的检测性能。**它的缺点是获得的特征不鲁棒，都是一些弱特征（因为很多的特征都是从较浅的层获得的）。讲了这么多终于轮到我们的FPN啦，它的架构如图d所示，整个过程如下所示，首先我们在输入的图像上进行深度卷积，然后对Layer2上面的特征进行降维操作（即添加一层1x1的卷积层），对Layer4上面的特征就行上采样操作，使得它们具有相应的尺寸，然后对处理后的Layer2和处理后的Layer4执行加法操作（对应元素相加），将获得的结果输入到Layer5中去。其背后的思路是为了获得一个强语义信息，这样可以提高检测性能。认真的你可能观察到了，这次我们使用了更深的层来构造特征金字塔，这样做是为了使用更加鲁棒的信息；除此之外，我们将处理过的低层特征和处理过的高层特征进行累加，这样做的目的是因为低层特征可以提供更加准确的位置信息，而多次的降采样和上采样操作使得深层网络的定位信息存在误差，因此我们将其结合其起来使用，这样我们就构建了一个更深的特征金字塔，融合了多层特征信息，并在不同的特征进行输出。这就是上图的详细解释。（个人观点而已）

# **FPN框架解析**

## **1. 利用FPN构建Faster R-CNN检测器步骤**

- 首先，选择一张需要处理的图片，然后对该图片进行预处理操作；
- 然后，将处理过的图片送入预训练的特征网络中（如ResNet等），即构建所谓的bottom-up网络；
- 接着，如图5所示，构建对应的top-down网络（即对层4进行上采样操作，先用1x1的卷积对层2进行降维处理，然后将两者相加（对应元素相加），最后进行3x3的卷积操作，最后）；
- 接着，在图中的4、5、6层上面分别进行RPN操作，即一个3x3的卷积后面分两路，分别连接一个1x1的卷积用来进行分类和回归操作；
- 接着，将上一步获得的候选ROI分别输入到4、5、6层上面分别进行ROI Pool操作（固定为7x7的特征）；
- 最后，在上一步的基础上面连接两个1024层的全连接网络层，然后分两个支路，连接对应的分类层和回归层；

![img](https://img-blog.csdn.net/20180309235331207)

图5 FPN整体架构

注：层1、2、3对应的支路就是bottom-up网络，就是所谓的预训练网络，文中使用了ResNet网络；由于整个流向是自底向上的，所以我们叫它bottom-up；层4、5、6对应的支路就是所谓的top-down网络，是FPN的核心部分，名字的来由也很简单。

作者的算法大致结构如下Fig3：**一个自底向上的线路，一个自顶向下的线路，横向连接（lateral connection）**。图中放大的区域就是横向连接，这里1*1的卷积核的主要作用是减少卷积核的个数，也就是减少了feature map的个数，并不改变feature map的尺寸大小。

![这里写图片描述](https://img-blog.csdn.net/20170606223026911?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDM4MDE2NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**自底向上**其实就是网络的前向过程。在前向过程中，feature map的大小在经过某些层后会改变，而在经过其他一些层的时候不会改变，作者将不改变feature map大小的层归为一个stage，因此每次抽取的特征都是每个stage的最后一个层输出，这样就能构成特征金字塔。
**自顶向下**的过程采用上采样（upsampling）进行，而横向连接则是将上采样的结果和自底向上生成的相同大小的feature map进行融合（merge）。在融合之后还会再采用3*3的卷积核对每个融合结果进行卷积，目的是消除上采样的混叠效应（aliasing effect）。并假设生成的feature map结果是P2，P3，P4，P5，和原来自底向上的卷积结果C2，C3，C4，C5一一对应。

##  **2. 为什么FPN能够很好的处理小目标？**

![img](https://img-blog.csdn.net/20180310092859891)

图6 FPN处理小目标

如上图所示，FPN能够很好地处理小目标的主要原因是：

- **FPN可以利用经过top-down模型后的那些上下文信息（高层语义信息）；**
- **对于小目标而言，FPN增加了特征映射的分辨率（即在更大的feature map上面进行操作，这样可以获得更多关于小目标的有用信息），如图中所示；**