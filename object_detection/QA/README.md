
1.二叉搜索树的插入与搜索，及其平均时间复杂度、最坏时间复杂度

2.二叉搜索树怎么转平衡二叉树  

3.C++的左值与右值，std:move()，深拷贝和浅拷贝

4.面向对象的概念

5.C++的虚函数

6.面向对象的三大特征

7.ROI Align的本质是不是resize操作？ROI Align细节

 

8.目标检测two-stage模型

​    RCNN->SPPNet-> Fast RCNN-> Faster RCNN-> RFCN->LibraRFCN-> DCN->DCNv2

9.目标检测one-stage模型

​    YOLOv1\v2\v3->SSD->RefineDet->retinaNet

​    YOLO和SSD的区别

10.resnet和densenet的区别，denseNet有没有改进模型（DPN、AAAI2018的MixNet），相同层数resnet、denseNet哪个好？

 

11.inception系列的演化

12.BN的原理和实现细节，其中均值和标准差的计算，以及训练和测试时分别怎么用

13.focal loss，用的什么核的，效果有没有区别，调过参数没有。Focal loss的两个参数有什么作用

14.小目标检测用什么方法

15.mobileNet v1\v2\v3

16.COCO冠军方案

17.多标签不平衡怎么处理，多任务不平衡怎么处理

18.改善NMS

​    IOU-guided-nms：IOUNet\softNMS\sofer-nms  实现细节

19.改善RPN

20.RFBNet

​    Receptive Field Block

​    模拟人类视觉的感受野加强网络的特征提取能力，在结构上RFB借鉴了inception的思想，主要是在inception的基础上假如了dilation卷积层，从而有效增大了感受野。

21.深度可分离卷积

22.two-stage为什么效果好

23.激活函数 [https://mlfromscratch.com/activation-functions-explained/#/](https://mlfromscratch.com/activation-functions-explained/#/)

24.损失函数，分类loss函数

25.数字图像处理的各种滤波

26.k折交叉验证

27.模型融合：adaboost

28.C++指针和引用的区别

29.mask rcnn和mask scoring rcnn

30.logistic回归、SVM、boosting/bagging

31.数据预处理transformer模块

32.处理不平衡的方法

33.圆上任意三个点组成的三角形包含圆心的概率

34.GAN

35.分布式，多卡使用

36.dataloder、dataset、sampler关系

37.人脸和身体一起检测，怎么处理

38.目标检测存在的问题，及个人理解；小目标怎么解决，遮挡怎么解决

39.概率：x,y,z都是（0,1）均匀分布，x+y+z<1的概率

40.人脸属性的任务、方法

41.n个文件（海量文件），查找和排序，二分查找的时间复杂度

42.知道哪些CV任务

​    分类、检测、分割、姿态、GAN、VAE\caption等

43.卷积、池化、全连接层、BN/IN/GN等

44.优化器

45.mAP的概念

46.传统机器学习，SVM\boosting\bagging\随机森林

​    Bagging和随机森林的区别

47.属性任务不平衡

48.属性任务的实际应用，目标检测实际应用

49.各种排序算法，快排时间复杂度，快排时间复杂度推导

50.时间复杂度为O(1)的排序算法

51.detection的two-stage的阈值有什么好方法，cascade RCNN?

52.模型具体的recall和precision

53.weighted sample 和focal loss

54.如果训练集不平衡，测试集平衡，直接训练、过采样和欠采样处理，哪个更好？

55.F1 score的alpha=1，那么alpha取其他值是什么含义。

56.canny边缘检测

57.SVM的核函数，损失函数

58.KNN

59.视频分类网络：分两路，一路提取视频音频特征，一路提取视频时空特征。如何融合两路特征。

60．Faster RCNN的细节，怎么筛选正负Anchor

61. OHEM原理

62. YOLO2细节

63.python的copy（）和deepcopy() 普通赋值，有啥区别

64.loss不降低怎么办，val loss不升（过拟合）怎么办

65.pytorch\tensorflow的区别

66. Discriminative loss 解释

67.模型压缩了解哪些

68.比赛的数据比例分别是多少，类别不平衡怎么处理

69.如何处理梯度弥散问题？CNN-LSTM

70.policy gradient和Q learning的区别

71.语义分割到实例分割怎么做

72.介绍下图像里面的多尺度（FPN、不同rate的空洞卷积等）

73.CTPN、OCR

74.SIFT、HOG。SIFT是如何保持尺度不变性的。

75.如何根据局部特征去检索更大的图片

76.OpenCV如何读取数据流中的图片

77.OpenCV如何生成图片

78.概率题：连续抛一枚公平的硬币，直到连续出现两次正面为止，平均要扔多少次硬币

79.交叉熵的公式，多类别交叉熵具体怎么计算的（标签x概率）

80.linux查找进程，查找文件

81.说说特征工程、特征融合原则、怎么筛选

82.boosting、lightGBM

83.介绍下ENet\UNet

84.实例分割、二值分割的区别

85.python的static装饰器

86.numpy用法，对某个矩阵的某一列全部置0，用什么操作

87.softmax公式，softmax的梯度是什么？

88.数据增强用了什么方法，在线增强和离线增强有什么区别

89.如何处理mask重合问题

90.faster rcnn中采用的类似focal loss的操作是什么

91.上采样方法、反卷积

92.Miou语义分割评价指标

93.两个bbox的顶点，如何快速判断重叠

94.说一下faster-rcnn的整个从输入到输出的框架流程

95.说一下RPN的原理

96.如何解决类内的检测

97.讲一下小目标检测，FPN为什么能提高小目标的‘准确率’，FPN的特征融合为什么是相加操作呢？FPN是怎么提高小目标的‘检出率’的？小目标在FPN的什么位置检测？

98.如果有很长、很小或者很宽的目标，用过如何处理

99.pytorch的卷积是如何实现的？

100.python多线程

101.大目标如果有两个候选框和GT重合应该怎么处理

102.为什么说ResNet101不适合检测

103．SENet为什么效果好？为什么说SENet泛化性能好？SE接在ResNet\inception的什么位置呢？

\104. sigmoid和softmax的区别

\105. DetNet原理

106.知识蒸馏：用大网络教小网络的方法？

107.假如一个图片中有一个很大的目标还有一个很小的目标，你会怎么处理？

108.多尺度训练如何设置？

109.长边为什么设置成1333，短边为什么设置成32的倍数？

110.anchor-free为什么能重新火起来？

111.smooth L1 loss为什么更有效？

112.SGD、Adam之类优化的原理

113.BN为什么有效？

114.python有哪些常用的库，报一遍

115.说一下使用pytorch对cifar10数据集分类的整个代码流程，构建模型的过程是怎么样的？

116.github的常用操作：上传、合并、分支之类的

117.linux的常用操作：查看文件大小、删除文件、查看文件行数、假如文件中的有很多文件，每个文件中又有很多文件，如何删除全部文件？

118.siamRPN、siamFC、DeSiam、SiamRPN++原理

119.有没有修改过轻量级模型、讲一下轻量级的模型。mobileNetv1\v2\v3，shuffleNetv1\v2、xception

120.决策树、集成学习

121.mask rcnn如何提高mask的分辨率

122.info GAN和GAN的细致区别

123.YOLO识别微笑物体效果差，为什么？

124.如何在3kw像素的图片中识别出10像素左右的瑕疵

125.mAP这个指标，在什么场景下适用，什么场景下会有问题，比如哪些问题？

126.WGAN的公式，原理

127.python的字典的实现

128.LR的损失函数与推导

129.C++ STL用过哪些，知道map的底层实现嘛？

130说说红黑树？

131.define和const和static区别？

132．与SGD类似的优化方法：momentum、Adagrad、Adam等

133.二阶优化方法有哪些？相比一阶的区别？

134.SVM推导。核函数，调参等

135.xgboost和gdbt怎么做回归和分类的，有什么区别？

136.c++虚函数？虚函数表？

137.python list反转？元素去重复？

138.depplab、ASPP是怎样的？

139.k-means是怎么实现的，k近邻算法呢？

140.静态变量有什么用，静态变量在哪初始化，能在类内初始化嘛？静态函数有什么用？

141.如何使用多线程加速pytorch的dataloader？

142.python的append和extend有什么区别？

143.BP的过程

144.反卷积具体怎么实现的？

145.进程与线程的区别，以及什么时候适合用线程进程

146.c++ STL中的map和hash_map的查找算法是怎么样的？时间复杂度是多少？

147.pytorch的permute和view的功能

148.手写计算AUC曲面面积的代码

149.如何解决过拟合

150.讲一下随机森林的原理

151.python的对象（object）和C++中的对象有什么区别？

152.python的lambda

153.手写中值滤波，介绍一下高斯滤波、均值滤波

154.进程、线程、协程的区别以及用处

155.c++中析构函数的作用、static的作用和特点

156.SIFT特征提取怎么做的，具备什么性质？为什么？ HOG特征提取怎么做的，具备什么性质？ Haar特征提取怎么做的，具备什么性质？LBP特征提取是怎么做的？
