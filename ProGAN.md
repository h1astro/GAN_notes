## ProGAN

>  notes written by h1astro
>
> 算力、数据、model->三大马车
>
> paper单位：NVIDIA
>
> ICLR 2018

#### 核心要点

1. 使用渐进的方式来训练生成器和判别器：先从生成**低分辨率**图像开始，然后不断增加**模型层数**来提升生成图像的细节
2. 方法能**加速**模型训练并大幅提升训练**稳定性**，生成前所未有的高质量图像**（1024*1024）**
3. 提出了一种简单方法来增加生成图像的**多样性**
4. 介绍了几种限制生成器和判别器之间**不健康竞争**的技巧
5. 提出了一种**评价**GAN生成效果的新方法，包括对生成质量和多样性的衡量
6. 构建了一个CELEBA数据集的**高清版本** HQ

 #### 研究背景

* 显示密度模型
  * 易解显性模型：定义一个方便计算的密度分布，主要的模型是Fully visible belief nets，简称FVBN，也被称作Auto-Regressive Network
  * 近似显性模型：可以定义任意的密度分布，使用近似方法来求解
* 隐性密度模型
  * GAN

![image-20210825194946588](/Users/shanjh/Library/Application Support/typora-user-images/image-20210825194946588.png)

* 神经自回归网络（PixelRNN/CNN）

通过链式法则把联合概率分布分解为条件概率分布的乘积 使用神经网络来参数化每个P

PixelRNN逐像素生成，效率很低，PixelCNN效果不如PixelRNN

* VAE-GAN

编码器：使P(z|x)逼近分布P(z)，比如标准正态分布，同时最小化生成器(解码器)和输入x的差距

解码器：最小化输出和输入x的差距，同时要骗过判别器

判别器：给真实样本高分，给重建样本和生成样本低分

![image-20210825195328882](/Users/shanjh/Library/Application Support/typora-user-images/image-20210825195328882.png)

##### GAN的损失函数

* f-divergence
  * JS散度(交叉熵)
  * LSGAN(MSE)
* Intergral probability metric。更好
  * **Wasserstein距离**

![image-20210825195541255](/Users/shanjh/Library/Application Support/typora-user-images/image-20210825195541255.png)

#### 图像生成的评价指标

* 可以评价生成样本的质量
* 可以评价生成样本的多样性，能发现过拟合、模式缺失、模式崩溃、直接记忆样本的问题
* 有界性，即输出的数值具有明确的上下界
* 给出的结果应当与人类感知一致
* 计算评价指标不应需要过多的样本
* 计算复杂度尽量低

![image-20210825195751982](/Users/shanjh/Library/Application Support/typora-user-images/image-20210825195751982.png)

Https://zhuanlan.zhihu.com/p/109342043

* Inception Score(IS)

  $exp(\frac{1}{N}\sum_{i=1}^N D_{KL}(p(y|x^{(i)})||\hat{p}(y)))$   KL散度越大越好

* Frechet Inception Distance(FID)。 表征距离，描述线的差异

  ![image-20210825200440087](/Users/shanjh/Library/Application Support/typora-user-images/image-20210825200440087.png)

* Maximum Mean Discrepancy (MMD)  迁移学习常用 最大均值差异

  ![image-20210825200428813](/Users/shanjh/Library/Application Support/typora-user-images/image-20210825200428813.png)

* MS-SSIM

  $SSIM(X,Y)=[L_M(X,Y)]^{\alpha M}{M}\underset{J=1}{\prod}^M[C_J(X,Y)]^{\beta_j}[S_J(X,Y)]^{{\gamma}_j}$

![image-20210825200922979](/Users/shanjh/Library/Application Support/typora-user-images/image-20210825200922979.png)

* 创建了首个大规模高清人脸数据库CelebA-HQ数据集，使得高清人脸生成的研究成为可能
* 首次生成了**1024*1024**分辨率的高清图像，确立了GAN在图像生成领域的**绝对优势**，大大加速了图像生成从实验室走向实际应用
* 从低分辨率逐次提升的策略缩短了训练所需的时间，训练速度提升2-6倍

![image-20210825201116748](/Users/shanjh/Library/Application Support/typora-user-images/image-20210825201116748.png)

#### 渐进式训练

* 生成器和判别器层数由浅到深，**不断增长**，生成图像的分辨率从4*4开始逐渐变大
* 生成器和判别器的增长保持**同步**，始终互为**镜像**结构
* 当前所有被添加进网络的层**都是可训练**的
* 新的层是平滑的添加进来，以防止对现有网络造成冲击

![image-20210826104011460](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826104011460.png)

* 新增加一个层时为过渡期，通过加权系数$\alpha$对上一层和当前层的输出进行加权
* $\alpha$从线性增长到1
* 在过渡期，判别器对真实图像和生成图像同样都进行$\alpha$加权
* 生成器中的上采样层使用最近邻Resize，判别器中的下采样使用平均赤化
* toRGB和fromRGB使用1*1卷积

![image-20210826104516844](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826104516844.png)

* 渐进式增长使训练更加稳定
* 为了证明渐进式增长与loss设计是正交的，论文中分别尝试了WGAN-GP和LSGAN两种loss
* 渐进式增长也能减少训练时间，根据输出分辨率的不同，训练速度能提升2-6倍
* WGAN-GP损失函数，使用gradient penalty策略来代替WGAN中的weight clipping，以使得判别器继续满足Lipschitz连续条件，同时判别器中无法再使用BN层

![image-20210826104937776](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826104937776.png)

#### Minibatch标准差

* 不需要任何参数或超参数
* 在判别器中，对于每个channel的每个像素点 分别计算batch内的标准差并取平均，得到一个代表整体标准差的标量。      [10,16,8,8]->[1,16,8,8]->[1,1,1,1]->[10,1,8,8]
* 复制这个标准差把它扩展为一个feature map，concat到现有维度上.  [10,16+1,8,8]
* 驾到判别器的**末尾**处效果最好
* 其它的一些增加生成多样性的方法，可以比这个方法效果更好，或者与此方法正交

### 归一化

#### 均衡学习率

##### He（Kaiming）初始化

目标：正向传播时，feature的方差保持不变；反向传播时，梯度的方差保持不变

* 适用于ReLU的初始化方法

  $W \sim N[0,\sqrt{\frac{2}{n_i}}]$

* 适用于Leaky ReLU的初始化方法：

  $W \sim N[0,\sqrt{\frac{2}{1+\alpha^2)\hat{n}_i}}]$

  $\hat{n}_i =h_i * w_i * d_i$

* 使用标准正态分布来初始化权重，然后在运行阶段对权重进行缩放，缩放系数使用He 初始化
* 之所以进行动态的缩放，而不是直接使用He初始化，与当前流星的**自适应随机梯度下降**方法（如Adam）中的尺度不变性相关
* 自适应随机梯度下降方法，会对频繁变化的参数以更小的步长进行更新，而稀疏的参数以更大的步长进行更新；比如在使用Adam时，如果某些参数的变化范围（标准差）比较大，那么它会被设置一个较小的学习速率
* 通过这样的动态缩放权重，在使用自适应随机梯度下降方法时，就可以确保所有权重的变化范围和学习速率都相同

#### 生成器中的像素归一化

* 希望能控制网络中的信号幅度

* 在省车更年期的每一个卷积层之后，对feature中**每个像素**在channel上归一化到单位长度

* 使用“局部响应归一化”对变体来实现

  $b_{x,y}=\alpha_{x,y} \sqrt{\frac{1}{N} \sum_{j=1}^{N-1} (a_{x,y}^j)^2 +\epsilon}$

* 一个**非常严格**对限制，不过却并没有让生成器的性能受到损失

* 对于大多数数据集而言，使用像素归一化后结果**没有太大变化**，但可以在网络的**信号强度过大时**进行有效抑制

### 评价指标

* MS-SSIM能发现GAN大尺度的模式崩溃，但对细节上颜色、纹理的多样性不敏感，并且也不能直接用评估两个图像数据集的相似性
* 作者认为，一个成功的生成器，生成的图像在任意尺度上，与训练集应该都有着良好的局部结构相似性
* 基于此设计了一种基于**多尺度统计相似性**的评价方法，来比较两个数据集的局部图像块之间的分布
* 随机选取了16384张图片，使用**拉普拉斯金字塔**抽取图像块，来进行图像的多尺度表达，尺寸从16*16开始，每次增大一倍一直到原始大小

![image-20210826112303344](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826112303344.png)

* 每个分辨率尺度上挑选128个描述子
* 每个描述子时一个**7x7x3的像素块**，3为颜色通道数
* 总共有16384*128=2.1M个大小为7x7x3=147的描述子
* 对每个描述子，在各个颜色channel上进行均值和标准差的**归一化**
* 使用sliced Wasserstein distance（SWD）来计算两组图像间各个**描述子的距离**
* SWD是一种对Wasserstein distance（推土机距离）的近似，因为两个高维分布间的WD不方便计算
* 比较小的SWD，表示两个图像数据集间的图像外观和整体方差都比较接近
* 对不同分辨率的SWD来说，16x16上的SWD代表**大尺度**上图像结构的相似性，而原始分辨率上的SWD则代表**像素级的差异**，比如噪声和边缘的锐度

![image-20210826114756562](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826114756562.png)

### 实验结果

#### 消融实验

* 生成图像的分辨率128x128，使用轻量型网络，在训练量达到10M时停止，网络还没有完全收敛
* MS-SSIM的评价使用了1w张生成图像
* 一开始batch size设为64，之后改为16
* 最终版本使用了更大的网络和更长时间的训练使得网络收敛，其生成效果至少可以与SOA相比较

![image-20210826115137371](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826115137371.png)

MS-SSIM越小越好

* 使用渐进式训练一方面可以提升生成质量，一方面可以减少训练的总时间
* 可以把渐进式的网络加深看作一种**隐式的课程学习**，从而来理解生成质量的提升

![image-20210826115553133](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826115553133.png)

#### 训练速度

* 使用渐进式训练后，低分辨率的SWD值很快就收敛，并在之后的训练中保持大致稳定
* 在生成1024分辨率的高清图像时，当训练量达到640万张图像时，渐进式训练要花费96个小时，而非渐进式训练经推断，大概需要520个小时，时渐进式训练的5.4倍。 如果是小分辨率 2-4倍吧

![image-20210826115824668](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826115824668.png)

#### CEKEBA-HQ

* celeba数据集有202599张图像，分辨率从43x55到6732x8984，不同图像质量差别很大
* 使用一个预训练的卷积**自编码器**来进行去除JPEG噪声
* 使用一个预训练的**4倍超分辨率GAN**来提升图像分辨率
* 基于CElebA中已有的脸部关键点标注，来进行人脸的截取和旋转矫正
* 处理了所有的CelebA图像，然后使用基于频谱的质量评价方式，选出**最好的3w张**生成图像

![image-20210826120256221](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826120256221.png)

* 使用8个Tesla V100 GPU并行训练了4天，此时SWD结果不再变卦
* 根据当前的训练分辨率，使用自适应的batch size，来最大效率的使用显存
* 为了证明作者的改进与loss很大程度上是相互独立的，分别尝试了LSGAN和WGAN-GP两种loss，LSGAN更不稳定但也能得到高清的生成图像
* 除了展示生成结果外，作者还进行了latent space的插值，和渐进式训练的可视化
* 插值方式：随机生成一系列latent code，然后对他们使用时域的高斯模糊，最后把各latent code归一化到一个超球面上

![image-20210826120614339](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826120614339.png)

#### CIFAR-10

* 在CIFAR-10上做图像生成，无监督的最高Inception score分数为7.90，有监督（带类别标签）的最高分数为8.87
* 有监督效果更好，因为无监督会因为不同类别图像间的过渡，而一定会生成**重影**的bad case，而带标签的训练能避免这一点
* 使用本文中的所有技巧后，在CIFAR10上的无监督训练能得到**8.80的IS分数**
* 此实验的大部分设置都与之间的CELEBA上的实验设置相同，除了WGAN-GP的超参数$\gamma$，由默认的1.0改成了750

![image-20210826120918507](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826120918507.png)

#### LSUN

![image-20210826120942227](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826120942227.png)

#### 讨论总结

##### 优点

* ProGAN相比于更早的GAN网络，生成的质量普遍都很高
* ProGAN在生成高分辨率的图像时也能够尽兴稳定的训练
* 目前的生成效果已经快令人信服了，特别是在CELEBA-HQ数据集上

##### 不足

* 离真正照片级的生成仍有很长一段距离。  如4096x4096
* 目前的生成还做不到对**图像语义与约束**的理解
* 生成图片的**细微结构**也还有改进的空间。    

![image-20210826121802112](/Users/shanjh/Library/Application Support/typora-user-images/image-20210826121802112.png)

### 论文总结

#### A 关键点

* 将高分辨率的图像生成任务进行分层，类似于一种课程学习，并且在课程之间进行平滑过渡
* 在学习率、多样性、抑制模式崩溃上都进行了改进，简单易用

#### B 创新点

* 完全镜像的网络
* 增加层数时的过渡
* 网络设计上力求简洁
* 新的SWD指标可以用来衡量和真实分布之间的差异，从而定量的指导训练

#### C 启发点

* 课程学习（渐进式）对于深度学习的意义
* 随着生成分辨率的增大，batch size成为了一个重要限制
* 根据实际需求选择合适的normalization
* 学习速率的影响不可小觑



#### 推荐资料       

**模型参数初始化总结**    
https://blog.csdn.net/qq_27825451/article/details/88707423

**优化器比较**     
https://www.cnblogs.com/guoyaohua/p/8542554.html
      
**其他能实现高清人脸生成的模型**    
VQ-VAE2:  https://www.jianshu.com/p/57bb6fdc143a   
GLOW:  https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/81039361?utm_medium=distribute.pc_relevant.none-task-blog-title-3&spm=1001.2101.3001.4242     
NVAE:  https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/107328328?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param    
ALAE:  https://www.cnblogs.com/king-lps/p/12796393.html      
