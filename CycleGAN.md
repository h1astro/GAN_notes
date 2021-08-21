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