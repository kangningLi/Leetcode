Inception 家族成员：Inception-V1（GoogLeNet）、BN-Inception、Inception-V2、Inception-V3、Inception-ResNet-V1、Inception-V4、Inception-ResNet-V2。

Inception系列网络结构可以模块化为：

𝐼𝑛𝑝𝑢𝑡→𝑆𝑡𝑒𝑚→𝐴→𝑅𝑒𝑑𝑢𝑐𝑖𝑡𝑜𝑛𝐴→𝐵→𝑅𝑒𝑑𝑢𝑐𝑡𝑖𝑜𝑛𝐵→𝐶→𝐴𝑣𝑔 𝑃𝑜𝑜𝑙𝑖𝑛𝑔(+𝐿𝑖𝑛𝑒𝑎𝑟)→𝑓𝑒𝑎𝑡𝑢𝑟𝑒

- **Stem**：前处理部分
- **A B C**：**网络主体**“三段式”，A B C每段的输入feature size依次折半，channel增加
- **ReductionA B**：完成feature size折半操作（降采样）
- **Avg Pooling (+ Linear)**：后处理部分

Inception系列的演化过程就是上面各环节不断改进（越来越复杂）的过程，其进化方向大致为

- **Stem**：大卷积层→多个小卷积层堆叠→multi-branch 小卷积层堆叠
- **A B C**：相同multi-branch结构→每阶段不同multi-branch结构→每阶段不同**Residual+multi-branch**结构，big convolution→ small convolution + BN → **factorized convolution**
- **ReductionA B**：max pooling → 不同multi-branch conv(stride 2)结构
- **后处理**：Avg Pooling + Linear → Avg Pooling



# Inception-V1 (GoogLeNet)

Inception-V1，更被熟知的名字为GoogLeNet，意向Lenet致敬。

通过增加网络深度和宽度可以提升网络的表征能力。

增加宽度可以简单地通过增加卷积核数量来实现，GoogLeNet在增加卷积核数量的同时，**引入了不同尺寸的卷积核，来捕捉不同尺度的特征**，形成了**multi-branch结构**——这是GoogLeNet网络结构的最大特点，如下图所示，然后将不同branch得到的feature map 拼接在一起，为了让feature map的尺寸相同，每个branch均采用SAME padding方式，同时**stride为1（包括max pooling）**。为了降低计算量，又**引入了1×11×1卷积层来降维**，如下图右所示，该multi-branch结构称之为一个Inception Module，在GoogLeNet中采用的是下图右的Inception Module。

[![source: http://arxiv.org/abs/1409.4842](https://s1.ax1x.com/2020/03/24/8L9wqI.png)](https://s1.ax1x.com/2020/03/24/8L9wqI.png)

直接增加深度会导致浅层出现严重的梯度消失现象，GoogLeNet引入了**辅助分类器（Auxiliary Classifier）**，在浅层和中间层插入，**来增强回传时的梯度信号，引导浅层学习到更具区分力的特征。**

GoogLeNet网络结构的特点可以概括为，

- 同时使用不同尺寸的卷积核，形成**multi-branch**结构，来捕捉不同尺度的特征
- **使用1×11×1卷积降维**，压缩信息，降低计算量
- 在classifier前使用**average pooling**





# BN-Inception

BN-Inception网络实际是在Batch Normalization论文中顺带提出的，旨在表现BN的强大。

[![source: http://arxiv.org/abs/1512.00567](https://s1.ax1x.com/2020/03/25/8jSbGT.png)](https://s1.ax1x.com/2020/03/25/8jSbGT.png)

与GoogLeNet的不同之处在于，

- 在每个激活层前**增加BN层**
- **将Inception Module中的5×5 卷积替换为2个3×3 卷积**，如上图所示
- 在Inception 3a和3b之后增加Inception 3c
- 部分Inception Module中的Pooling层改为average pooling
- 取消Inception Module之间衔接的pooling层，而将下采样操作交给Inception 3c和4e，令stride为2



# Inception-V2, V3

Inception V2和V3出自同一篇论文[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)。

GoogLeNet和BN-Inception网络结构中Inception Module可分为3组，称之为3x、4x和5x（即主体三段式A B C），GoogLeNet和BN-Inception这3组采用相同Inception Module结构，只是堆叠的数量不同。

Inception V2和V3与以往最大的不同之处在于3组分别使用了不同结构的Inception Module，分别如下图从左到右所示，

[![source: http://arxiv.org/abs/1512.00567](https://s1.ax1x.com/2020/03/25/8jioCt.png)](https://s1.ax1x.com/2020/03/25/8jioCt.png)

具体地，

- 3x使用的Inception Module与BN-Inception相同，即将5×5拆分成2个堆叠的3×3 ；
- 4x使用的Inception Module采用了factorized convolutions ，**将2维卷积拆分成2个堆叠的1维卷积，可类比传统计算机视觉中的“行列可分解卷积”，但中间夹了个激活**，1维卷积的长度为7；
- 5x使用的Inception Module，1维卷积不再堆叠而是并列，将结果concat；

除此之外，

- 3x和4x之间，4x和5x之间，均不存在衔接的池化层，下采样通过Inception Module中的stride实现
- 取消了浅层的辅助分类器，只保留中层的辅助分类器
- 最开始的几个卷积层调整为多个堆叠的3×3 卷积



据论文所述，V3与V2的差异在于，

- RMSProp Optimizer
- **Label Smoothing**，**训练中使用的label为one hot label与均匀分布的加权**，可以看成一种正则
- Factorized 7×7，即将第一个7×7卷积层变为堆叠的3个3×3
- BN-auxiliary，辅助分类器中的全连接层也加入BN

但是，**实际发布的Inception V3完全是另外一回事**，参见[pytorch/inception](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py)，有人绘制了V3的网络架构如下——网上少有绘制正确的，下图中亦存在小瑕疵，最后一个下采样Inception Module中1×11×1的stride为1。

需要注意的是，起下采样作用两个Inception Module并不相同。



# Inception-V4，Inception-ResNet-v1，Inception-ResNet-v2

Inception-V4，Inception-ResNet-v1 和 Inception-ResNet-v2出自同一篇论文[Inception-V4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)，

Inception-V4相对V3的主要变化在于，**前处理使用更复杂的multi-branch stem模块**，主体三段式与V3相同。

[![https://towardsdatascience.com/review-inception-v4-evolved-from-googlenet-merged-with-resnet-idea-image-classification-5e8c339d18bc](https://s1.ax1x.com/2020/04/01/G3gdq1.png)](https://s1.ax1x.com/2020/04/01/G3gdq1.png)

Inception-ResNet-V1与Inception-ResNet-V2，将Inception与ResNet结合，**使用Inception结构来拟合残差部分**，两者在A B C部分结构相同，只是后者channel数更多，两者的主要差异在前处理部分，后者采用了更复杂的multi-branch stem结构（与V4相同）。相比纯Inception结构，**引入ResNet结构极大加快了网络的收敛速度**。

[
](https://s1.ax1x.com/2020/04/01/G3TWpq.png)

