## Pix2Pix

#### 摘要

1. 研究条件GAN网络在图像翻译任务中的通用解决方案
2. 网络不仅学习到从输入图像到输出图像的映射，还学习用于训练该映射的损失函数
3. 证明了这种方法可以有效应用在图像合成、图像上色等多种图像翻译任务上
4. 使用作者发布的pix2pix软件，大量用户已经成功进行了自己的实验，进一步证明了方法的泛化性
5. 表明可以在不手工设计损失函数的情况下，也能获得理想的结果

#### 研究背景

数字图像任务：

* 计算机视觉

  模仿人眼和大脑对视觉信息的处理和理解。

  图像分类，目标检测，人脸识别

* 计算机图像学

  在数字空间中模拟物理世界的视觉感知

  动画制作，3D建模，虚拟现实

* 数字图像处理

  依据先验知识，对图像的展现形式进行转换

  图像增强，图像修复，相机ISP

![image-20210816091943054](/Users/shanjh/Library/Application Support/typora-user-images/image-20210816091943054.png)

#### 图像翻译

图像与图像之间以不同形式的转换。根据source domain的图像生成target domain中的对应图像，约束生成的图像和source图像的分布在某个维度上尽量一致

* 图像修复
* 视频插帧
* 图像编辑
* 风格迁移
* 超分辨率

#### 图像质量评价 （image Quality Assessment, IQA)

有参考图像损失

* 像素损失。   MSE -> PSNR （加了log）
* 结构性损失 SSIM
* 色彩损失
* 锐度损失. GMSD
* 感知损失.  

![image-20210816092519324](/Users/shanjh/Library/Application Support/typora-user-images/image-20210816092519324.png)

#### 数据集

CMP Facade Database



Paris StreetView Dataset



CItyscapes Dataset

![image-20210816092745422](/Users/shanjh/Library/Application Support/typora-user-images/image-20210816092745422.png)



#### 研究成果

![image-20210817093729292](/Users/shanjh/Library/Application Support/typora-user-images/image-20210817093729292.png)

![image-20210817093744155](/Users/shanjh/Library/Application Support/typora-user-images/image-20210817093744155.png)

#### 研究意义

给出了图像翻译任务的通用框架，对于不同类型的任务，不需要一个特定的算法和损失函数

使GAN从此统治了各类图像翻译任务，促使各类图像翻译任务的效果逐渐接近了实用级别



#### 目标函数

 Loss由传统像素loss和GAN loss组成

L1 loss比L2 loss更能减少生成图像的模糊

初始层的随机噪声z，在训练过程中会被忽略，导致网络的条件输入只对应固定的输出

在推断时，通过保留dropout来产生轻微的随机性

如何得到更大随机性的生成结果，更完整的去拟合输入的分布，仍然有待研究

$ \mathcal L_{cGAN}(G,D)=\mathbb E_{x,y}[\log{D(x,y)}]+\mathbb E_{x,z}{[\log{(1-D(x,G(x,z)))}]}$

$ \mathcal L_{GAN}(G,D)=\mathbb E_y{[\log{D(y)}]} + \mathbb E_{x,z}[\log{(1-D(G(x,z)))}]$

$ \mathcal L_{L1}(G)=\mathbb E_{x,y,z}[||y-G(x,z)||_1]$

$G^*=arg\underset{G}{min} \underset{D}{max} \mathcal  L_{cGAN}(G,D)+\lambda \mathcal L_{L1}(G)$



### 生成器

##### Unet

基于经典的Encoder-decoder结构

在很多图像翻译任务中，输出和输出图像外观看起来不同，但结构信息是相同的

在Encode过程中，feature map的尺寸不断减小，低级特征将会丢失

在第i层和第n-i层间加入skip-connection，把i层的特征直接传到第n-i层

##### PatchGAN

* 像素级的L1 loss能很好的捕捉到图像中的低频信息，GAN的判别器只需要关注高频信息

* 把图像切成N*N的patch，其中N显著小于图像尺寸

* **假设在大于N时**，像素之间是相互独立的，从而可以把图像建模成马尔可夫随机场

* 把判别器在所有patch上的推断结果，求平均来作为最终输出

* 可以把PatchGAN理解为对图像 纹理/style损失的计算

* PatchGAN具有较少的参数，运行得更快，并且可以应用于任意大的图像



#### 训练参数

* 采用原始GAN的loss替换技巧，把最小化log(1-D(x,G(x,z)))，替换成最大化logD(x,G(x,z))
* 生成器和判别器的训练次数为1:1，训练速度为2:1，稍微降低判别器的优化速度
* 使用Adam优化器，学习速率为2e-4
* Batch size在1-10之间，不同的数据集上有所区别
* 推断时保留dropout，来增加输出的随机性
* 推断时使用batch为1的batch normalization，实际就是instance normalization
* instance normalization仅在图像内部进行操作，可以保留不同生成图像之间的独立性 

