## Improved Techniques for Training GANs

> notes written by h1astro
>
> 1. 结构改进和训练技巧
> 2. 用于半监督学习
> 3. 生成图像的质量评价
> 4. 代码实现

#### 核心要点：

提出了 一系列新的GAN结构和训练方式

进行了半监督学习和图像生成相关实验

新的技术框架在MNIST、CIFAR-10和SVHN的半监督分类中取得了良好效果

通过视觉图灵测试证明，生成的图像同真实图像已难以区分

在ImageNet上训练，模型学习到了原图的显著特征

#### 数据标签

* 监督学习：数据有完整的标签
* 无监督学习：没有数据标签
  * 自监督学习：自动生成标签
* 半监督学习：少量数据拥有完整标签
* 弱监督学习：数据有**不完整**的标签
* 强化学习：一开始没有标签，从环境中逐渐生成标签

![image-20210807100621778](/Users/shanjh/Library/Application Support/typora-user-images/image-20210807100621778.png)



半监督学习：

流形假设：将高维数据嵌入到低维流形中，当两个样例位于低维流形中的一个小局部邻域内时，具有相似的类标签

![image-20210807101045877](/Users/shanjh/Library/Application Support/typora-user-images/image-20210807101045877.png)

#### 研究意义：

在DCGAN的基础上，通过多种正则化手段提升可收敛性

综合质量和多样性，给出了一条评价GAN生成效果的新路径

展示了把GAN生成图像，作为其它图像任务训练集的可行性