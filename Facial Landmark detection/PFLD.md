## PFLD

 Our model can be merely 2.1Mb of size and reach over 140 fps per face on a mobile phone (Qualcomm ARM 845 processor)

More concretely, we customize an end-to-end single stage network associated with acceleration techniques. During the training phase, for each sample, rotation information is estimated for geometrically regularizing landmark localization, which is then NOT involved in the testing phase. A novel loss is designed to, besides considering the geometrical regularization, mitigate the issue of data imbalance by adjusting weights of samples to different states, such as large pose, extreme lighting, and occlusion, in the training set. Extensive experiments are conducted to demonstrate the efficacy of our design and reveal its superior performance over state-ofthe-art alternatives on widely-adopted challenging benchmarks, i.e., 300W (including iBUG, LFPW, AFW, HELEN, and XM2VTS) and AFLW

1. 局部变化: 现实场景中人脸的表情、光照、以及遮挡情况变化都很大;
2. 全局变化: 姿态和成像质量也影响图像中人脸的表征，人脸全局结构的错误估计直接导致定位不准;
3. 数据不平衡: 不平衡的数据使得算法模型无法正确表示数据的特征;
4. 模型的有效性: 由于手机和嵌入式设备计算性能和内存资源的限制，必须要求检测模型的size小处理速度快;

**data imbalance**

To address this issue, we advocate to penalize more on errors corresponding to rare training samples than on those to rich ones.

稀有的 more error

novel loss： Considering the above two concerns, say the **geometric constraint** and the **data imbalance**

To enlarge the receptive field and better catch the global structure on faces, a **multi-scale fully-connected (MS-FC) layer is added** for precisely localizing landmarks in images. 

 **As for the processing speed and model compactness,** we build the **backbone network of our PFLD using MobileNet blocks**

#### Loss Function(感觉和focal loss思想类似)

**算法思想**

作者使用的网络结构如下：

![img](http://5b0988e595225.cdn.sohucs.com/images/20190303/a2d6d0336471443788eeedce31bf7a07.jpeg)

其中，

黄色曲线包围的是主网络，用于预测特征点的位置；

绿色曲线包围的部分为辅网络，在训练时预测人脸姿态（有文献表明给网络加这个辅助任务可以提高定位精度，具体参考原论文），这部分在测试时不需要。

**作者主要用两种方法，解决上述问题。**

对于上述影响精度的挑战，修改loss函数在训练时关注那些稀有样本，而提高计算速度和减小模型size则是使用轻量级模型。

- Loss函数设计

Loss函数用于神经网络在每次训练时预测的形状和标注形状的误差。

考虑到样本的不平衡，作者希望能对那些稀有样本赋予更高的权重，这种加权的Loss函数被表达为：

![img](http://5b0988e595225.cdn.sohucs.com/images/20190303/81c83f02b52b4157800923166d7af0b7.png)

M为样本个数，N为特征点个数，Yn为不同的权重，|| * ||为特征点的距离度量（L1或L2距离）。(以Y代替公式里的希腊字母)

进一步细化Yn:

![img](http://5b0988e595225.cdn.sohucs.com/images/20190303/9b575afa7bff4714aa26e596cc186594.jpeg)

其中

![img](http://5b0988e595225.cdn.sohucs.com/images/20190303/93db3b2739ee4d3980705f75c6efffd2.png)

即为最终的样本权重。

K=3，这一项代表着人脸姿态的三个维度，即yaw, pitch, roll 角度，可见角度越高，权重越大。

C为不同的人脸类别数，作者将人脸分成多个类别，比如侧脸、正脸、抬头、低头、表情、遮挡等，w为与类别对应的给定权重，如果某类别样本少则给定权重大。

![img](https://img-blog.csdn.net/20140915160335655?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd3NqOTk4Njg5YWE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

yaw，pitch，roll： 分别代表上下翻转，左右翻转，平面内旋转的角度。

![1](/Users/likangning/Desktop/1.png)

三组 multi scale fully connected layer 结束后是把feature map给concatenate起来。

![2](/Users/likangning/Desktop/2.png)

![3](/Users/likangning/Desktop/3.png)

It is easy to obtain that PC c=1 ω c n PK k=1(1 − cos θ k n ) in Eq. (2) acts as γn in Eq. (1). Let us here take a close look at the loss. In which, θ 1 , θ 2 , and θ 3 (K=3) represent the angles of deviation between the ground-truth and estimated yaw, pitch, and roll angles. Clearly, as the deviation angle increases, the penalization goes up. In addition, we categorize a sample into one or multiple attribute classes including profile-face, frontal-face, head-up, head-down, expression, and occlusion. The weighting parameter ω c n is adjusted according to the fraction of samples belonging to class c (this work simply adopts the reciprocal of fraction). For instance, if disabling the geometry and data imbalance functionalities, our loss degenerates to a simple `2 loss. No matter whether the 3D pose and/or the data imbalance bother(s) the training or not, our loss can handle the local variation by its distance measurement.