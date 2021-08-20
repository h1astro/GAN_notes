## Pix2Pix

> notes written by h1astro
>
> 如果创新度不够，可以多角度实验，提高泛用性

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

#### 人工评价

Amazon Mechanical Turk （AMT）

任务：地图生成，航拍照片生成，图像上色

* 限制观看时间，每张图像停留一秒钟，答题时间不限
* 每次评测只针对一个算法，包含50张图像
* 前10张图像为练习题，答题后提供反馈，后40张为正式标注
* 每批评测数据由50个标注者标注，每个评测者只能参与一次评测
* 评测不包含测验环节



#### FCN-score

Fully Convolutional Networks

* 判别图像的类别是否能被模型识别出来，类似于Inception Score
* 使用应用于语义分割任务的流行网络结构-- FCN-8s，并在cityscapes数据集上进行训练
* 根据网络分类的精度，来对图像的质量进行打分
* 由于图像翻译不太关注生成图像的多样性，所以不需要像Inception Score 一样关注总体生成图像的分布

![image-20210820095654014](/Users/shanjh/Library/Application Support/typora-user-images/image-20210820095654014.png)

#### 目标函数分析

* 只使用L1 loss，结果合理但模糊
* 只使用cGAN loss，结果锐利但存在artifacts
* 通过对L1 loss和cGAN loss进行一定的加权（100:1），可以结合两者的优点
* 只使用GAN loss，生成图像发生模式崩溃
* 使用GAN+L1，结果于CGAN+L1类似

![image-20210820100046970](/Users/shanjh/Library/Application Support/typora-user-images/image-20210820100046970.png)

![image-20210820100057154](/Users/shanjh/Library/Application Support/typora-user-images/image-20210820100057154.png)

* 光谱维度中的情况，与像素维度傻姑娘的情况类似
* 只使用L1 loss，更倾向于生成平均的浅灰色；使用cGAN loss，生成的色彩更接近真实情况

![image-20210820100158999](/Users/shanjh/Library/Application Support/typora-user-images/image-20210820100158999.png)

#### 模型分析

##### 生成器分析

Encoder-Decoder VS U-net

在两种loss下，U-net的效果都显著优于Encoder-Decoder

![image-20210820100524785](/Users/shanjh/Library/Application Support/typora-user-images/image-20210820100524785.png)

##### 判别器分析

* 比较不同的Patch大小的差异，默认使用L1+cGAN loss

* 1\*1的patch相比于L1 loss，只是增强了生成图像的色彩使用16\*16的patch，图像锐度有所提升，但存在显著的tiling artifacts
* 使用70\*70的patch，取得了最佳效果
* 使用286\*286的完整图像输入，生成结果比70\*70的要差,因为网络的参数量大很多，训练更难以收敛

#### 应用分析

##### Map-Photo

Map to Photo， L1+ cGan loss的结果显著优于L1 loss

photo to map，L1+cGan loss的结果接近于只使用L1 loss

原因分析：可能是Map相比对Photo，几何特征要显著的多

![image-20210820101559670](/Users/shanjh/Library/Application Support/typora-user-images/image-20210820101559670.png)

![image-20210820101608023](/Users/shanjh/Library/Application Support/typora-user-images/image-20210820101608023.png)

##### 图像上色

L1+cGan loss的结果，与只使用L2 loss的结果较为接近

相比于此前专门针对图像上色问题的方法，效果仍有较大差距

![image-20210820101710451](/Users/shanjh/Library/Application Support/typora-user-images/image-20210820101710451.png)

![image-20210820101721635](/Users/shanjh/Library/Application Support/typora-user-images/image-20210820101721635.png)

##### 语义分割

* 只使用cGAN loss时，生成的效果和模型准确度还可以接受
* 使用L1+cGAN loss，效果反而不如只使用L1 loss
* 说明对于CV领域的问题，传统loss可能就够了 加入GAN loss 无法获得增益

![image-20210820101843437](/Users/shanjh/Library/Application Support/typora-user-images/image-20210820101843437.png)

![image-20210820101850520](/Users/shanjh/Library/Application Support/typora-user-images/image-20210820101850520.png)

#### 结论

* 条件对抗网络在许多图像翻译任务上都非常有应用潜力
* 特别是在高度结构化图像输出的任务上
* 使用不同的数据集训练pix2pix，可以用于各种图像翻译任务

#### 论文总结

##### 关键点

* 把语义分割任务的最新成果，与cGAN结合进行推广
* 把像素级loss与GAN loss相结合
* 在各类图像翻译任务上进行丰富实验

##### 创新点

* 大视野，针对一个广泛的应用场景
* 用传统loss处理低频信息，GAN loss处理高频信息
* 对图像处理与CV任务进行联合分析

##### 启发点

* 对已有的成果推广其应用范围，也能产生很大的价值
* 简洁往往是泛用性的前提条件
* 如果研究创新度不够，就从其它角度挖掘亮点

