## StackGAN

> notes written by h1astro
>
> ICCV 2017
>
> 条件增强、两阶段的GAN
>
> 将高分辨率的图像生成任务进行分解
>
> 基于VAE的思想+多级网络，使用了条件增强技术
>
> 但还称不上照片级的图像，以及选取的花和鸟背景过于简单

#### 核心要点

1. 现有文本到图像方法生成的样本，可以大致表达出给定的文本含义，但是图像细节和质量不佳
2. StackGAN能基于描述，生成**256x256**分辨率的照片级图像。（还不能和ProGAN比）
3. 把问题进行了分解，采用草图绘制-精细绘制 **两阶段** 过程
4. 阶段1的GAN根据给定的文本描述，来绘制对象的原始形状和颜色；阶段2度GAN使用**文本描述和阶段1的输出**来作为输入，通过纠正草图中的缺陷和细节生成，来最终得到更高分辨率的图像
5. 还提出了一种条件增强方法，能够增强**潜在条件流行的平滑性** （latent space 不过是从词向量得来）
6. 大量实验表明，以上方法再以文本描述为条件的照片级图像生成上取得了显著进步

#### 研究背景

##### Energy-Based（EB）GAN

* 将判别器视为一个**energy function**，函数值（**非负**）越小代表data越可能是真实数据
* 使用**自编码**作为判别器（energy function）
* 判别器可以单独使用真实数据进行提前的**预训练**
* 可以基于ImageNet数据集训练，生成256x256分辨率的图拍呢

$D(x)=||Dec(Enc(x))-x||$

$\mathcal L_D(x,z)=D(x)+[m-D(G(z))]^+$    不需要大于m，超过变0？

$\mathcal L_G(z)=D(G(z))$

![image-20210828092220455](/Users/shanjh/Library/Application Support/typora-user-images/image-20210828092220455.png)

![image-20210828092231342](/Users/shanjh/Library/Application Support/typora-user-images/image-20210828092231342.png)

##### 变分自编码器（VAE）

* 希望根据离散变量集合【X1，···，Xn】得到X的原始**分布**p(X)

  $P(X)=\int P(X|z;\theta)P(z)dz$

* 假设P(z)~N(0,I)

* 对于大部分z，$P(X|z;\theta)$接近0，不太好求解

* 改为计算$p(Z|X)$

* 使用**变分推断**把对$p(Z|X)$进行**近似求解**

  $\mathcal D[Q(z)||P(z|X)]=E_{z \sim Q}[\log{Q(z)}-\log{P(z|X)}]$

![image-20210828092858186](/Users/shanjh/Library/Application Support/typora-user-images/image-20210828092858186.png)

* 上述优化项经推导可改写为

  $E_{z \sim Q}[\log{P(X|z)}-\mathcal D[Q(z|X)||P(z)]]$

* P(X|z)通过Decoder来学习

* 令Q(z|X)为**高斯分布**，通过Encoder来学习

* 为了可导，从Q(z|X)采样时，使用**重参化技巧**

![image-20210828093312958](/Users/shanjh/Library/Application Support/typora-user-images/image-20210828093312958.png)

![image-20210828093322518](/Users/shanjh/Library/Application Support/typora-user-images/image-20210828093322518.png)

##### 文本生成图像

* VAE

* DRAW（Deep Recurrent Attention Writer)

  * 使用循环神经网络+注意力机制
  * 依次生成一个个对象叠加在一起得到最终结果

  ![image-20210828093740435](/Users/shanjh/Library/Application Support/typora-user-images/image-20210828093740435.png)

* GAN

  * 在生成器中，text embedding跟随机噪声融合后一起输入到生成网络中
  * 鉴别器会对错误情况进行分类，一种生成的法克图像匹配了正确的文本，另一种是真实图像但匹配了错误文本

  ![image-20210828093811566](/Users/shanjh/Library/Application Support/typora-user-images/image-20210828093811566.png)

#### 研究成果

* 首次在文本到图像的任务中，生成了256x256分辨率的高质量图像
* 提出的条件增强方法， 能增强模型的鲁棒性病提升生成效果的多样性

 ![image-20210828094059436](/Users/shanjh/Library/Application Support/typora-user-images/image-20210828094059436.png)

#### 研究意义

* 成为了文本生成图像任务中的一个里程碑
* 基于VAE思想的条件增强方法，对之后的研究者造成了一定启发

![image-20210828094148391](/Users/shanjh/Library/Application Support/typora-user-images/image-20210828094148391.png)



#### 条件增强

* 本文嵌入的隐空间维度通常**非常高**（>100），在数据量有限的情况下，通常会导致隐数据流形中的**不连续性**
* 从高斯分布$\mathcal N(\mu(\varphi_i),\sum(\varphi_i))$中**随机采样**latent code，其中$\mu(\varphi_i)$和$\sum(\varphi_i)$是关于词向量$\varphi_i$的函数
* 均值$\mu$和方差$\sum$使用一个**全连接层**
* 把$D_{KL}(\mathcal N(\mu(\varphi_i),\sum(\varphi_i)||\mathcal N(0,I))$作为一个**正则项**加入生成器的训练
* 使用**重参化**技巧，通过$\hat{c}_0=\mu_0+\sigma_0 \odot \epsilon$进行采样$\epsilon \sim \mathcal N(0,I)$
* 使用上述的条件增强方法后，可以产生**更多的训练数据**，使条件流形更加平滑
* 增加的采样随机性，可以使输入同一个句子时产生**不同的输出图像**

![image-20210829093331866](/Users/shanjh/Library/Application Support/typora-user-images/image-20210829093331866.png)

类似VAE-GAN

#### 两阶段的GAN

##### 阶段一

* 从**标准高斯分布**中采样得到z，与从条件增强方法采样得到的$\hat{c}_0$进行concat作为输入
* $I_0$为文本描述所对应的真实图像，在所有实验中**$\lambda$都设为1**
* 在判别器中，输入图像经过下采样，最终得到长宽为M的矩阵；而词向量会先经过全连接层来**压缩到N维**，然后在空间维度上复制变为MxMxN的矩阵
* 图像和词向量的矩阵concat到一起，再通过1x1卷积和全连接层得到最终的输出分数

![image-20210829093857683](/Users/shanjh/Library/Application Support/typora-user-images/image-20210829093857683.png)

##### 阶段2

* 把阶段1的输出$s_0=G_0({z,\hat{c}_0})$与又一次条件增强采样得到的$\hat{c}$进行concat作为输入
* 在生成器中增加了**残差block**；判别器中的**负样本**有真实图像-错误文本，生成图像-正确文本两种情况

![image-20210829094536286](/Users/shanjh/Library/Application Support/typora-user-images/image-20210829094536286.png)

和阶段一区别，s0变了

![image-20210829094544001](/Users/shanjh/Library/Application Support/typora-user-images/image-20210829094544001.png)

#### 实现细节

* 上采样使用最近邻resize+3x3卷积
* 除了最后一层外，在每个卷积层之后都是用了BN和ReLU
* 在128x128的StackGAN中使用了2个残差block，在256x256中使用了4个
* 判别器中，下采样的block使用4x4步长为2的卷积，除了第一层没使用BN外，别的都使用了BN和LeakyReLU
* 首先训练阶段1的GAN 600个epochs，接着将其固定，再训练阶段2的GAN 600个epochs
* 都使用Adam优化器，batch size设为64
* 初始学习率设为2e-4，之后进行指数衰减，每100个epochs衰减到1/2

#### 评价方式

#### CUB200-2011（Caltech-UCSD Birds）

* 加州理工学院在2010年提出，包含11788张鸟类图像，有200个不同的鸟类子类，每张图像那个均提供了标记信息，包含鸟的bounding box，312个属性信息和15个关键part信息
* 对CUB的图像进行裁剪，以使得物体-图像的尺寸比例大于0.75

#### Oxford-102

* 由牛津大学工程科学系在2008年发布，包含8189张jpg格式的花朵图像，有102类产自英国的花卉，每类有40-258张图片

#### 客观评价

* 使用Inception Score

  $I=exp(\mathbb E_x D_{KL}(p(y|x)||p(y)))$

* COCO数据集上，直接使用预训练的Inception模型

* 对于CUB和Oxford-102，使用**finetune**后的Inception模型

#### 主观评价

* 从COCO的验证集中随机选择4k个文本描述
* 从CUB和Oxford-102测试集中随机选择50个文本描述
* 对于每个描述，使用模型生成**5个图像**
* 在相同的文本描述下，10个评测者对不同模型输出的结果进行排名

#### 模型比较

* 对于CUB、Oxford-102和COCO三个数据集，StackGAN在客观和主观评价上都取得了最佳结果
* GAN-INT-CLS只能生成64x64分辨率的图像，缺乏图像细节，得分较低
* GAWWN虽然可以取得更高的得分，但需要使用**额外的输入信息**（位置等），否则无法取得任务有意义的输出，并且得分仍然低于StackGAN

![image-20210829100822921](/Users/shanjh/Library/Application Support/typora-user-images/image-20210829100822921.png)

* 阶段1的GAN能够绘制对象的粗略形状和颜色，但一般模糊不清，缺少细节并带有各种缺陷，特别是对于前景目标
* 阶段2的GAN会在阶段1的基础上进一步补充细节，并且在阶段1**没能绘制合理的形状时**，仍然能生成合理的对象
* 使用阶段2的判别器来踢去生成图像和真实图像的特征，以寻找离生成图像最接近的真实图像， 结果表明，生成的结果**并不是简单的复制真实图像**

阶段1 64x64，阶段2 256x256

![image-20210829101111433](/Users/shanjh/Library/Application Support/typora-user-images/image-20210829101111433.png)

#### 组件分析

* 计算IS得分之前，图片会被resize到**299x299**
* 使用一阶段的GAN，把生成分辨率从64增大到256，IS得分反而变得更低了，说明如果不能为图像增加更多的细节，简单的增大输出尺寸并不能提高得分
* 使用两阶段的GAN，128x128分辨率的得分低于256x256，说明在输出分辨率增大的同时，确实**细节得到了提升**
* 使用**条件增强**后，训练更加稳定，IS得分的**显著的提升**，生成图像也具备了多样性，因为这为隐条件流形带来了小的随机扰动，从而增强了模型的鲁棒性
* 在两个阶段的GAN都**输入词向量**，IS得分显著低于只在阶段1输入词向量

![image-20210829101609959](/Users/shanjh/Library/Application Support/typora-user-images/image-20210829101609959.png)

* 为了证明StackGAN最终得到了一个平滑的隐条件流形，进行了latent code的线形插值实验
* 分别进行了颜色的插值，和多种属性的**联合插值**
* 插值结果显示，生成图像的属性也是进行着**渐变**，并且大致具有类似的图像质量，表示隐条件流形**较为平滑**

> 实际效果还不是很好，不能说达到照片级

![image-20210829101814294](/Users/shanjh/Library/Application Support/typora-user-images/image-20210829101814294.png)

#### 结论

* 提出了堆叠的GAN网络，结合条件增强方法， 能够生成照片级的图像（？ 选的图片没啥背景，存疑）

> 人脸应该更难点

* 其中阶段1的GAN网络根据给定的文本描述，来生成颜色和形状基本满足要求的草图
* 阶段2的GAN网络，能够纠正阶段1结果的缺陷，并增加更多细节
* 一系列实验显示，与现有的文本到图像方法相比，StackGAN能生成具有更高分辨率更多细节和多样性的目标图像 

![image-20210829102441012](/Users/shanjh/Library/Application Support/typora-user-images/image-20210829102441012.png)

#### 论文总结

#### A 关键点

* 将高分辨率的图像生成任务进行分解
* 结合VAE，使用了条件增强技术

#### B 创新点

* VAE-GAN +多级网络

#### C 启发点

* 结合不同生成式模型的混合模型，在某些特定领域存在用武之地

