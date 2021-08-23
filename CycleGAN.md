## CycleGAN

> notes written by h1astro

#### 核心要点

1. 图像翻译任务 需要**对齐**的图像对，但很多场景下无法获得这样的训练数据
2. 提出了一个基于非配对数据的方法，仍然可以学习到不同domain图像间的映射
3. CycleGAN是在GAN loss的基础上加入**循环一致性损失**，使得F(G(X))尽量接近X (反之亦然)
4. 在训练集没有配对图像的情况下，对CycleGAN在风格迁移、物体变形、季节转换、图像增强等多个图像翻译任务中的生成结果做了定性展示
5. 与此前一些方法的定量比较，进一步显示了CycleGAN的优势

#### 研究背景

双射（Bijection）

既是单射又是满射的映射，即“一一映射”

* 映射：两个非空集合X与Y间存在着对应关系f，而且对于X中的每一个元素x，Y中总有唯一的一个元素y与x对应
* 单射（injection）：对于X中不同的元素x，其所对应y也各不相同
* 满射（surjection）：对于Y中的每一个元素y，都至少存在一个x与其对应

![image-20210821160446765](/Users/shanjh/Library/Application Support/typora-user-images/image-20210821160446765.png)

#### 域自适应/泛化 （Domain Adaptation/Generalization）

* domain自适应/泛化是迁移学习的一块重要研究领域
* 不同形式和来源的数据，其domain各不相同，数据分布存在域差异（domain discrepancy）
* 而domain自适应/泛化的目标，就是学习到不同domain间的域不变（domain invariant）特征

#### 神经风格迁移（Neural Style Transfer）

* 在CNN中，通过CV任务学习到的content表征和style表征可以进行区分
* Gram矩阵：一组向量间两两的内积所组成的矩阵，称为这组向量的Gram matrix，它可以表示这组向量的**长度**以及之间的**相似程度**
* 使用预训练CNN模型中的高层feature作为输入图像的content表征，使用各层feature的Gram矩阵作为输入的style表征

![image-20210821161416637](/Users/shanjh/Library/Application Support/typora-user-images/image-20210821161416637.png)

#### 研究成果

![image-20210821161556394](/Users/shanjh/Library/Application Support/typora-user-images/image-20210821161556394.png)

![image-20210821161642562](/Users/shanjh/Library/Application Support/typora-user-images/image-20210821161642562.png)

花的背景虚化

#### 研究意义

*  在pix2pix的基础上，通过引入循环一致性，减少了对配对数据集的需求，进一步拓展了GAN在图像翻译领域的应用范围
* 进一步验证了GAN+循环一致性/对偶（dual）的思路，可以在无监督Domain Adaptation的领域中取得不错的效果

#### 目标

* 在X和Y两个不同domain间，建立起**双向**的映射关系G和F；并使用两个判别器Dx和Dy，来分别对{x}和{F(Y)}、{y}和{G(x)}进行区分
* 对抗损失——使得映射后的数据分布接近目标domain的数据分布
* 循环一致性损失——保证学习到的两个映射G和F不会相互矛盾 

![image-20210822122828778](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822122828778.png)

#### 对抗损失

使用的是和传统GAN网络一致的对抗损失函数

$\mathcal L_{GAN}(G,D_Y,X,Y)=\mathbb E_{y\sim p_{data}(y)}[\log{D_Y(y)}]+\mathbb E_{x\sim p_{data}(x)}[\log{(1-D_Y(G(x)))}]$

优化目标也是两个min-max函数

$\underset{G}{min} \underset{D_Y}{max} \mathcal L_{GAN}(G,D_Y,X,Y)$

$\underset{F}{min} \underset{D_X}{max} \mathcal L_{GAN}(F,D_X,Y,X)$

#### 循环一致性损失

* 如果只使用对抗损失，理论上当网络足够大时，模型有能力将输入图像集合映射到**任意**一组图片集合，所以无法保证输入能被正确映射到对应的输出图像

* 如：正常情况下，在斑马->野马 两组图像间进行翻译时， 对应关系应当如下：

  站姿斑马->站姿野马。   跪姿斑马->跪姿野马

  但如果映射发生交叉，即：

  站姿斑马->跪姿野马。  跪姿斑马->站姿野马

  两种情况的对抗损失没有区别，训练时无法区分模型是否学习到了所期望的映射关系

* 为了避免产生错误的交叉映射，引入了循环一致性损失

* 对于任意一个x和y，应该有：$x \rarr G(x) \rarr F(G(x)) \approx x$

  ​													$y \rarr F(y) \rarr G(F(y)) \approx y$

* 使用L1距离时，则损失函数为：

  $\mathcal L_{cyc}(G,F)=\mathbb E_{x\sim p_{data}(x)}[||F(G(x))-x||_1]+\mathcal E_{y\sim p_{data}(y)}[||G(F(y))-y||_1]$

* 实验中发现，使用对抗损失来代替x和F(G(X))之间的L1距离，并不能带来性能上的提升

* 后续实验中，展示了重建的图像F(G(x))与输入图像x确实非常接近

* 综上所述，得到的完整损失函数如下：

  $\mathcal L(G,F,D_X,D_Y)=\mathcal L_{GAN}(G,D_Y,X,Y)+\mathcal L_{GAN}(F,D_X,Y,X)+ \lambda \mathcal L_{cyc}(G,F)$

  $G^*,F^*= arg\underset{G,F}{min}\underset{D_x,D_Y}{max} \mathcal L(G,F,D_X,D_Y)$

* 整个模型可以被看作是两个“自编码器” （auto-encoder）

* 这两个“自编码器”需要学习到符合另一个domain的隐层表达

* 更准确的说，它们是“对抗自编码器”（Adversarial Auto-Encoder）的一个特例

* 实验中，对loss各成分的重要性进行了进一步分析

#### 网络结构

* 生成器使用《perceptual lossses for real-time style transfer and super-resolution》

  中的图像翻译网络，该模型在神经风格迁移和超分辨率中已经取得了很好的效果

* 输入图像分辨率为128\*128,使用6个残差block；分辨率为256\*256及以上时，使用9个残差block

* 生成器中使用了Instance Normalization

* 判别器使用Patch大小为70\*70的PatchGAN

![image-20210822125929882](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822125929882.png)

#### 训练细节

* 使用最小二乘损失（least squares loss）取代原来的最大似然估计损失，这可以提升训练的稳定性和生成图像的质量

  we train the G to minimize $\mathbb E_{x\sim p_{data}(x)}[(D(G(x))-1)^2]$ 

  train the D to minimize $\mathbb E_{y\sim p_{data}(y)}[(D(y)-1)^2]+ \mathbb E_{x\sim p_{data}(x)}[D(G(x))^2]$

* 为了减小训练的震荡，不使用最近一张图片（batch）来更新判别器，而是最近的50张图片

* 权重$\lambda$设为10

* 使用Adam优化器

* Batch size设为1

* 学习率在前100个epoch中为2e-4，在接下来的100个epoch中将线性递减到0

### 模型评价

#### 评估指标

* AMT评估，地图->航拍照片的翻译任务，每个参与算法的评测有25个标注员参与，其它都与pix2pix中的人工评测流程一致
* 因为评价标准略有差异，不能直接与pix2pix中国呢的结果比较，并且标注员的来源分布也不同）两次实验的时间不一样
* 使用FCN score来对Cityscapes的语义标签->照片翻译任务进行评估
* 使用语义分割的评价方法来评估 照片->语义标签的翻译任务，本文使用CItyscapes的benchmark，包括像素准确率、类别准确率和类别的交并比（IOU）

#### 基础模型 baseline models

* CoGAN

  在两个domain上分别训练一个GAN的生成器，这两个生成器共享前几层权重；这样以来，使用生成图像X对应的latent code，就能通过另一个生成器生成与X存在对应关系的Y

* SImGAN

  在GAN loss的基础上加入了$||x-G(x)||_1$的loss

* Feature loss+GAN

  SImGAN的一种变体，使用感知loss(VGG-16 relu4_2)来代替原有的RGB像素级loss

* BiGAN/ALI

  除了训练从Z到X的生成器之外，还训练了从X到Z的反向映射，本文使用target图像Y来代替Z

* Pix2pix

  基于配对图像训练，作为非配对模型效果的一个“上界”

除了CoGAN使用开源实现外，其它baseline都采用与CycleGAN相同的网络结构和训练参数 ![除了CoGAN使用开源实现外，其它baseline都采用与CycleGAN相同的网络结构和训练参数](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822131717887.png)

* 在AMT的人工评测中，CycleGAN有1/4左右的概率能欺骗评测员，而其它baseline几乎完全无法欺骗评测员（看结果确实..很差）

* 在三类评测任务重，CycleGAN各项得分都显著优于所有的baseline模型

![image-20210822132053060](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822132053060.png)

![image-20210822132105818](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822132105818.png)

![image-20210822132114286](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822132114286.png)

#### 损失函数分析。消融实验

* 只使用单独的GAN损失、或者只使用单独的循环一致性损失时，最终的结果都很差
* 在GAN损失的基础上，只添加单项的循环一致性损失时，训练很容易不稳定并发生模式崩溃，特别是在不包含映射方向的循环一致性损失时

![image-20210822132702005](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822132702005.png)

![image-20210822132815875](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822132815875.png)

GAN alone和GAN+forward出现了模式崩溃，两幅图一样了

#### 图像重构质量

* 在训练和测试中，重构图像与原始图都很接近
* 对于domain差距较大的图像翻译任务一样，比如地图和航拍图的相互转换

区别在于亮度、色彩

![image-20210822150717289](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822150717289.png)

#### 集合风格迁移

* 训练数据来自Flickr和WikiArt
* 四种风格画作的数据集大小分布是526，1073，400和563
* CycleGAN可以生成梵高风格的画作，神经风格迁移只能模仿单幅画如《星夜》的风格

![image-20210822150916546](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822150916546.png)

* 为了比较神经风格迁移和集合风格迁移的效果，基于神经风格迁移方法，计算**平均Gram矩阵**，以此来学习整个集合的风格，并使用平均Gram矩阵进行“平均神经风格迁移”
* 这样处理方式，经常无法生成真实效果的图片

#### 物体变形

* 作者在ImageNet上不同种类的物体之间训练CycleGAN，每种物体使用1000张图片训练
* CycleGAN能够完整不同种类物体之间的变形，只要这些物体看起来是相似的
* 之前Turmukhambetov等人提出基于子空间的方法只能在同类物体中进行变形

![image-20210822151200746](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822151200746.png)

#### 季节转换

数据来自Flickr，包含约塞米蒂国家公园的854冬季照片和1273张夏季照片

![image-20210822151258413](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822151258413.png)

### 从画作生成照片

* 作者加入了一个新的损失函数$L_{identity}$，使网络更倾向于保留原有的颜色组成
* 不使用$L_{identity}$时，生成器经常把白天时的绘画变成日落时的照片
* 其它地方展示都为测试集结果，但此应用展示训练集结果，因为大师们无法再有新作品了，不需要考虑泛化性

![image-20210822151533757](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822151533757.png)

#### 照片增强

* 从Flick上下载分别由小光圈智能手机和大光圈单反相机拍摄的花的照片作为训练集
* CycleGAN成功的把智能手机拍摄的照片转换为了小景深照片，实现了背景虚化效果

![image-20210822151658229](/Users/shanjh/Library/Application Support/typora-user-images/image-20210822151658229.png)

#### 局限与结论

* CycleGAN在物体变形任务上表现较差，这可能与生辰器的结构相关，是未来需要重点改进的方向
* 数据的分布特征会对网络性能产生影响，由于训练集中没有人骑马的数据，导致测试时出现了“斑马人”
* 在配对图像数据集和非配对图像数据集上分别训练的模型，其能力之间还时有**难以弥补**的差距
* 通过加入弱监督/半监督学习，有可能大大提升CycleGAN的效果
* CycleGAN拓宽了无监督方法可以处理的任务范围

#### 论文总结

##### A 关键点

* 构造了两个生成对抗网络，分别进行对抗性训练
* 在损失函数中加入了循环一致性损失

##### B 创新点

* 结合使用残差网络、PatchGAN、循环一致性损失
* 将GAN成功应用在多个非配对图像翻译任务中

##### C 启发点

* 一个简单而古老的idea，依靠故事好、实验多、易复现，同样能脱颖而出
* 无监督学习模型拥有非常广泛的应用范围，存在巨大的研究价值

#### 推荐资料       

Deep Domain Adaptation论文集   
https://zhuanlan.zhihu.com/p/53359505    

Neural Style Transfer 神经风格迁移详解    
https://blog.csdn.net/Cowry5/article/details/81037767

ECCV 2020 ｜ 基于对抗一致性的非匹配图像转换     
https://zhuanlan.zhihu.com/p/156092551?utm_source=wechat_session  