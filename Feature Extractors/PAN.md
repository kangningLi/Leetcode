**Path Aggregation Network (PANet)** aiming at boosting information flow in proposal-based instance segmentation framework.we enhance the entire feature hierarchy with accurate localization signals in lower layers by **bottom-up path augmentation**, which **shortens the information path between lower layers and topmost feature**

We present **adaptive feature pooling**, which links feature grid and all feature levels to make useful information in each feature level propagate directly to following proposal subnetworks. **A complementary branch capturing different views for each proposal is created to further improve mask prediction**.

是基于mask rcnn的改进

**features in low levels are helpful for large instance identification**. But there is a long path from low-level structure to topmost features, increasing difficulty to access accurate localization information.

#### MaskRcnn 的缺点：

1. low level的特征对大目标的识别很有帮助，但是从low level 到 topmost 的特征中间有很长的一个path，这样就加大了对物体精确的位置信息获取的准确率。
2.  each proposal is predicted based on feature grids pooled from one feature level, which is assigned heuristically.  此分配是启发式的， 这个过程可以被update因为在别的层被discard掉的特征可能对最后的定位是有帮助的。
3. mask prediction is made on a single view, losing the chance to gather more diverse information.

#### 改进

1. **bottom-up path augmentation**：to shorten information path and enhance feature pyramid with accurate localization signals existing in low-levels, 

2. **Adaptive pooling layer**：  to recover broken information path between each proposal and all feature levels
3.  **to capture different views of each proposal**, we augment mask prediction with **tiny fully-connected (fc) layers**, which possess complementary properties to FCN originally used by Mask R-CNN. 全连接融合层：使用一个小型fc层用于补充mask预测

![img](https://leanote.com/api/file/getImage?fileId=5d82e196ab6441313d0003d6)



#### Augmented Bottom-up Structure

**Figure1是关于PANet的示意图，主要包含FPN、bottom-up path augmentation、adaptive feature pooling、fully-connected fusion四个部分。**
**FPN**是CVPR2017的一篇文章，主要是通过融合高低层特征提升目标检测的效果，尤其可以提高小尺寸目标的检测效果。
**Bottom-up Path Augmentation**的引入主要是考虑网络浅层特征信息对于实例分割非常重要，这个也非常容易理解，毕竟浅层特征多是边缘形状等特征，而实例分割又是像素级别的分类。那么为什么bottom-up path augmentation能保留更多的浅层特征呢？作者在Figure1中用红绿两个箭头来解释了。**红色虚线箭头**表示在FPN算法中，因为要走自底向上的过程，浅层的特征传递到顶层要经过几十甚至一百多个网络层（在FPN中，对应Figure1中那4个蓝色矩形块从下到上分别是ResNet的res2、res3、res4和res5层的输出，层数大概在几十到一百多左右），显然经过这么多层的传递，浅层特征信息丢失会比较厉害。**绿色虚线箭头**表示作者添加一个bottom-up path augmentation，本身这个结构不到10层，这样浅层特征经过底下原来FPN的lateral connection连接到P2再从P2沿着bottom-up path augmentation传递到顶层，经过的层数就不到10层，能较好地保留浅层特征信息。关于bottom-up path augmentation的具体设计参考后面的Figure2，最后融合得到的特征层是N2、N3、N4、N5，其中N2和P2相同，这些特征层用于后续的预测框分类、回归和mask生成。
**Adaptive Feature Pooling**主要做的还是特征融合。我们知道在Faster RCNN系列的目标检测或分割算法中，RPN网络得到的ROI需要经过ROI Pooling或ROI Align提取ROI特征，这一步操作中每个ROI所基于的特征都是单层特征（FPN也是如此），比如ResNet网络中常用的res5的输出。而adaptive feature pooling则是将单层特征也换成多层特征，也就是说每个ROI需要和多层特征（文中是4层）做ROI Align的操作，然后将得到的不同层的ROI特征融合在一起，这样每个ROI特征就融合了多层特征。具体内容可以看后面的Figure6。
**Fully-connected Fusion**是针对原有的分割支路（FCN）引入一个前背景二分类的全连接支路，通过融合这两条支路的输出得到更加精确的分割结果。具体内容可以看后面的Figure4。





**Figure2是bottom-up path augmentation的示意图**，是比较常规的特征融合操作，比如Ni经过卷积核尺寸为3*3，stride=2的卷积层，特征图尺寸缩减为原来的一半，然后和Pi+1做element-wise add，得到的结果再经过一个卷积核尺寸为3*3，stride=1的卷积层得到Ni+1，特征图尺寸不变。
![这里写图片描述](https://img-blog.csdn.net/20180729173635259?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQzODAxNjU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**Figure6是adaptive feature poolng的示意图**。RPN网络得到的每个ROI都要分别和N2、N3、N4、N5层特征做ROIAlign操作，这样每个ROI就提取到4个不同的特征图，然后将4个不同的特征图融合在一起就得到最终的特征，后续的分类和回归都是基于最终的特征进行。
![这里写图片描述](https://img-blog.csdn.net/20180729173708197?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQzODAxNjU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

之所以引入adaptive feature pooling其实是基于FPN中提取ROI特征的思考，虽然FPN网络基于多层特征做预测，但是每个ROI提取特征时依然是基于单层特征，然而单层特征就足够了吗？于是作者做了Figure3这个实验，Figure3中有4条曲线，对应FPN网络中基于4层特征做预测，每一层都会经过RPN网络得到ROI，所以这4条曲线就对应4个ROI集合。横坐标则表示每个ROI集合所提取的不同层特征的占比。比如蓝色曲线代表level1，应该是尺度比较小的ROI集合，这一类型的ROI所提取的特征仅有30%是来自于level1的特征，剩下的70%都来自其他level的特征，leve2、leve3、leve4曲线也是同理，这说明原来RPN网络的做法（level x的ROI所提取的特征100%来自于leve x的特征，x可取1、2、3、4）并不是最佳的。因此就有了特征融合的思考，也就是每个ROI提取不同层的特征并做融合，这对于提升模型效果显然是有利无害。

![这里写图片描述](https://img-blog.csdn.net/20180729173728231?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQzODAxNjU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)







**Figure4是fully-connected fusion的示意图**。主要是在原来的mask支路（Figure4上面那条支路，也就是传统的FCN结构）上增加了Figure4下面那条支路做融合。增加的这条支路包含2个3*3的卷积层（其中第二个为了降低计算量还将通道缩减为原来的一半），然后接一个全连接层，再经过reshape操作得到维度和上面支路相同的前背景mask，也就是说下面这条支路做的是前景和背景的二分类，因此输出维度类似文中说到的28*28*1。上面这条支路，也就是传统的FCN结构将输出针对每个类别的二分类mask，因此输出的通道就是类别的数量，输出维度类似28*28*K，K表示类别数。最终，这两条支路的输出mask做融合得到最终的结果。因此可以看出这里增加了关于每个像素点的前背景分类支路，通过融合这部分特征得到更加精确的分割结果。
![这里写图片描述](https://img-blog.csdn.net/20180729173746727?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQzODAxNjU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)