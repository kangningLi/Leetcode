A **1 x 1 Convolution** is a convolution with some special properties in that it can be used for **<u>dimensionality reduction</u>**, **<u>efficient low dimensional embeddings, and applying non-linearity after convolutions.</u>** It maps an input pixel with all its channels to an output pixel which can be squeezed to a desired output depth. It can be viewed as an [MLP](https://paperswithcode.com/method/feedforward-network) looking at a particular pixel location.

那么1×1卷积核有什么作用呢，如果当前层和下一层都只有一个通道那么1×1卷积核确实没什么作用，但是如果它们分别为m层和n层的话，**1×1卷积核可以起到一个跨通道聚合的作用**，**所以进一步可以起到降维（或者升维）的作用，起到减少参数的目的**。
比如当前层为 x×x×m即图像大小为x×x，特征层数为m，然后如果将其通过1×1的卷积核，特征层数为n，那么只要n<m这样就能起到降维的目的，减少之后步骤的运算量（当然这里不太严谨，需要考虑1×1卷积核本身的参数个数为m×n个）。如果使用1x1的卷积核，这个操作实现的就是多个feature map的线性组合，可以实现feature map在通道个数上的变化。
而因为卷积操作本身就可以做到各个通道的重新聚合的作用，所以1×1的卷积核也能达到这个效果。

**增加非线性特性**

1*1卷积核，可以在保持feature map尺度不变的（即不损失分辨率）的前提下大幅增加非线性特性（利用后接的非线性激活函数），把网络做的很deep。