## S 3FD: Single Shot Scale-invariant Face Detector

其实呢，这篇文章对我最大的启发，是发现问题，然后解决问题。目前大多数anchor-based的detector，都存在检测小物体的时候，performance就急剧下降了，作者首先分析了一下原因。

![img](https://pic1.zhimg.com/80/v2-dee59f1d8873a328af04e44360e5985d_720w.jpg)

- 最低的anchor-associated layer的stride size太大了，拿VGG16的conv5_3来说，stride是16，就是原图一个16*16的人脸，到这一层直接被压缩成1个pixel了，损失了大量的信息；作者的解决方法还是利用了多层的信息，SSD只利用了conv4 3，作者还用了conv5 3以及conv33，当然用之前还normalize了一下，解决一下数值不均衡的问题。
- 对于小的人脸，anchor scale以及receptive field是相互不匹配的，导致anchor实际上不能很好的fit人脸，于是这个就需要针对不同层的receptive field去单独设计anchor scale。
- 传统SSD的做法在matching的时候把每个anchor匹配的最好的ground truth叫做besttruthoverlap, 然后看这个overlap是否大于一个阈值，大的话就是face，否则就是background，但是这样会出现一个问题，很多ground truth对应的anchor都是负的，这太unbalance了，作者就觉得不行，从矮子里挑将军，我必须给你找个positive的样本出来，于是提出了一个两阶段的匹配策略，在下文中我们也会详细说到。
- 为了检测小的人脸，我们会设计很多小的anchor，这就导致negative anchor数量的提升，是一个unbalance的二分类问题，于是作者这边就在conv3_3层提出了使用max-out background label。因为这一层的anchor scale是最小的。

所以其实看到思路是一目了然的，先看目前的detector有什么问题，然后去解决它。

### Scale-equitable Framework

![img](https://pic3.zhimg.com/80/v2-356fc954432135636d0e8d4fab09d1fe_720w.jpg)

这边是使用了conv3_3, conv4_3, conv5_3，然后这些层的数值还不一样，还加了一个L2 Normalize。把他们的norm限制到10，8还有5。额外还是用了conv_fc7, conv6_2, conv7_2。这是所有的检测层了。每个检测层后面都会follow一个px3x3xq的卷积层，p是input的channel，q是output的channel。对于每个anchor，预测坐标的4个偏移量，以及 ![[公式]](https://www.zhihu.com/equation?tex=N_s) 个分类分数， ![[公式]](https://www.zhihu.com/equation?tex=N_s+%3D+N_m+%2B+1) ， ![[公式]](https://www.zhihu.com/equation?tex=N_m) 是maxout background label，这个只是对conv3_3层的，对于其他的层， ![[公式]](https://www.zhihu.com/equation?tex=N_s+%3D+2) 。

关于为人脸的不同的detection layer设计不同的anchor，首先aspect ratio设置为了1：1，因为人脸的bouding box近似为方的。 可以看到这个表1，每一层的stride以及感受野是固定的，这也是设计anchor scale的两个base point。

- 有效的感受野，不是感受野内的每个pixel都是同样重要的，总的来说，中央的神经元起到的作用是比四周的大的，所以anchor应该是要比感受野小的来匹配有效感受野。
- detection layer的stride大小决定了anchor在input image上的间隔。以conv3_3为例，stride是4，anchor是16*16这就表明了在原图上每隔4个pixel有一个16*16的anchor。我们就把这个anchor的scale设计成了stride的4倍，这样就能保证，不同scale的anchor在原图上有同样的密度。这样不同scale的脸能尽可能匹配到同等数目的anchor。

在训练的时候我们还要判断，哪个anchor与一个face有关，anchor的scale是离散的，但是人脸的scale是连续的。这边引入了两个阶段的匹配策略，首先，把SSD中的阈值0.5降低到0.35，先粗糙的匹配一下，即使这样还是有一些face没有anchor匹配，我们就选择和这些face的iou>0.1的anchor，排序完之后选择topN个，作为match这些face的anchor。看一下第一阶段，每个face平均匹配到多少anchor，这边N就设置为整个平均数。

这篇文章的最后一个trick就是这个max-out label了。在SFD中，anchor-based人脸检测方法，是一个二分类问题，来决定是否是face还是background。但是这个anchor的分类是一个unbalanced问题，只有极小的一部分是positive。比如说，一个640x640的image，总共有34125个anchor，有大约75%的来自于conv3_3，这也是与最小锚点相关的层（16x16）。这些小的anchor对于那些false positive的样本做出了极大的贡献。因此，通过平铺小的anchor来提升小面孔的检出率，必然会导致小面孔的false positive rate（把不是人脸的看成人脸）。为了解决这问题，在这个conv3_3的detection layer上应用了max-out background label，对于背景预测了个 ![[公式]](https://www.zhihu.com/equation?tex=N_m) 分数，选择最高的作为final score

## 代码参考

https://zhuanlan.zhihu.com/p/78923965

