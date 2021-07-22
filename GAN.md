GAN

* 联合概率 P(Y|X)

* 条件概率 P(X,Y)

判别式模型

* 模型学习的是条件概率分布P(Y|X),

* 任务是从属性X（特征）预测标记Y（类别）
* 只关心符合条件一部分的数据

生成式模型

* 模型学习的是联合概率分布P(X,Y)
* 任务是得到属性为X且类别为Y时的联合概率
* 全局上的分布，更难以训练

判别式模型

* 线性回归
* 逻辑回归
* K近邻
* 支持向量机
* 决策树
* 条件随机场
* boosting方法

生成式模型

* 朴素贝叶斯
* 混合高斯模型
* 隐马尔可夫模型 HMM
* 贝叶斯网络
* 马尔可夫随机场
* 深度信念网络DBN
* 变分自编码器

Minimax

* 在零和博弈中，为了使己方达到最优解，所以把目标设为让对方的最大收益最小化

数据集

* MNIST 
  * 手写数字集，源自NIST，28*28的灰度图，train 6w，test 1w
  * Yann.lecun.com/exdb/mnist

* TFD
  * The Toronoto face dataset 人脸数据集
* CIFAR-10
  * 32*32 彩图，10个类别，每类6k张，train 5w，test 1w
  * www.cs.toronto.edu/~kriz/cifar.html

GAN历史意义

* 使AI具备了图像视频的创作编辑能力
* 启发了无/弱监督学习的研究

GAN应用领域

* 图像生成
* 图像转换
* 图像编辑

 ![image-20210720090348754](/Users/shanjh/Library/Application Support/typora-user-images/image-20210720090348754.png)



VAE (Variational Auto-Encoder)

编码器

把数据编码成mean vector 和standard deviation vector

采样

从构建的高斯分布中采用得到latent vector

解码器

从latent vector 生成数据

![image-20210720090952704](/Users/shanjh/Library/Application Support/typora-user-images/image-20210720090952704.png)

#### 价值函数

$\underset{G}{min}\underset{D}{max}V(D,G)=\mathbb{E}_{x\sim p_{data}(x)}[\log{D(x)}]+ \mathbb{E}_{z \sim p_z{(z)}}\log{(1-D(G(z)))}]$

* Data :真实数据
* D： 判别器，输出值为[0,1]，代表输入来自真实数据的概率
* z： 随机噪声
* G：生成器，输出为合成数据

#### D的目标，是最大化价值函数V

最大化V，就是最大化D(x)和1-D(G(z))

对于任意的x，都有D(x)=1,

对于任意的z,都有 D(G(z))=0

#### 训练流程

使用mini-batch梯度下降（带momentum）

训练k次判别器（论文实验中k=1）

训练1次生成器

![image-20210720093217603](/Users/shanjh/Library/Application Support/typora-user-images/image-20210720093217603.png)

#### 全局最优解

判别器最优解：$D_G^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$

判别器取得最优解时，$p_g=p_{data}$

生成器的最优解: $C^* = -\log{4}$

信息熵：

* $H(U)=E[-\log{p_i}]=-\sum_{i=1}^n p_i \log{p_i}$

* 一个概率分布的复杂程度

KL散度:

* $D_{KL}(P||Q)=-\sum P(x)\log{1/P(x)}+ \sum P(x)\log{1/Q(x)} = \sum P(x) \log{P(x)}/Q(x)$
* 基于分布Q描述分布P所需的额外信息量
* P和Q差别的非对称性的度量

JS散度

* $JS(P_1 ||P_2)=1/2KL(P_1||(P_1+P_2)/2)+ 1/2 KL(P_2|| (P_1+P_2)/2)$
* 基于KL散度，解决了KL散度**非对称的问题**。     $P_1 <- \quad \frac{P_1+P_2}{2}\quad ->P_2$
* 如果两个分布距离较远 没有重叠部分时， KL散度没有意义，而JS散度为常数1

#### 生成器的可收敛性

凸函数（convex function）

局部最优解等于全局最优解

上确界（supremum）

一个集合的最小上界，与最大值类似

次导数（subderivatives）

做一条直线通过点(x，f(x))，并且要么接触f，要么在f的下方，这条直线的斜率称为f的次导数

次梯度算法（subgradient method）

与梯度下降算法类似，用次梯度代替梯度，在凸函数上能确保收敛性



#### 模型优劣势

缺点：

* 没有显示表示的$P_g(x)$
* 必须同步训练G和D，可能会发生。**模式崩溃**

优点：

* 不使用马尔可夫链，在学习过程中不需要推理
* 可以将多种函数合并到模型中
* 可以表示**非常尖锐**、**甚至退化**的分布
* 不是直接使用数据来计算loss 更新生成器，而使用判别器的梯度，所以数据**不会直接复制**到生成器的参数中

#### 论文总结

关键点

* 对抗性框架
* 价值函数的设计
* 全局最优解
* GAN的改进方向

创新点

* 使用神经网络来判断两个分布的相似程度
* 把两个相互对抗的loss作为唯一的优化目标

启发点

* 从生物智能中继续挖掘宝藏
* 对抗在机器学习中的应用，更多的对抗方式
* 深度神经网络handle一切
* 评价俩个分布的相似性是一个重难点
* 能否结合统计学方法与DL方法的优缺点



### 问题：

* 【思考题】虽然理论上GAN的全局最优解和可收敛性已经得到证明，但实践上还可能存在哪些问题，该如何解决？

​	根据价值函数 V(G,D) 的定义，需要求两个数学期望，即 E[log(D(x))] 和 E[log(1-D(G(z)))]，其中 x 服从真实数据分布，z 服从初始化分布。但在实践中，没有办法利用积分求这两个数学期望的，所以一般从无穷的真实数据和无穷的生成器中做采样以逼近真实的数学期望。

​	必须使用迭代和数值计算的方法实现极小极大化博弈过程。在训练的内部循环中完整地优化 D 在计算上是不允许的，并且有限的数据集也会导致过拟合。因此我们可以在 k 个优化 D 的步骤和一个优化 G 的步骤间交替进行。那么我们只需慢慢地更新 G，D 就会一直处于最优解的附近，这种策略类似于 SML/PCD 训练的方式。

**Mode collapse(模型崩溃)**
Mode collapse 是指 GAN 生成的样本单一，其认为满足某一分布的结果为 true，其他为 False，导致以上结果。
自然数据分布是非常复杂，且是多峰值的(multimodal)。也就是说数据分布有很多的峰值(peak)或众数(mode)。每个 mode 都表示相似数据样本的聚集，但与其他 mode 是不同的。
在 mode collapse 过程中，生成网络 G 会生成属于有限集 mode 的样本。当 G 认为可以在单个 mode 上欺骗判别网络 D 时，G 就会生成该 mode 外的样本。

**Convergence(收敛)**
GAN 训练过程中遇到的一个问题是什么时候停止训练？因为判别网络 D 损失降级会改善生成网络 G 的损失(反之亦然)，因此无法根据损失函数的值来判断收敛，



* 【代码实践】对提供的现有代码进行完善，加入模型保存、模型推断代码

```python
    from torchvision.utils import save_image

    epoch = 0 # temporary
    batches_done = epoch * len(dataloader) + i
    if batches_done % opt.sample_interval == 0:
        save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True) # 保存生成图像
        
        os.makedirs("model", exist_ok=True) # 保存模型
        torch.save(generator, 'model/generator.pkl') 
        torch.save(discriminator, 'model/discriminator.pkl')
        
        print("gen images saved!\n")
        print("model saved!")
```



* 【总结】搜索了解其它生成式模型，如VAE、自回归模型等，自行总结GAN与其它生成式模型等异同

**1) 生成对抗网络（GAN）**。

**2) 变分自动编码模型（VAE）**。它依靠的是传统的概率图模型的框架，通过一些适当的联合分布的概率逼近，简化整个学习过程，使得所学习到的模型能够很好地解释所观测到的数据。

**3) 自回归模型（Auto-regressive）**。简单认为，每个变量只依赖于它的分布，只依赖于它在某种意义上的近邻。例如将自回归模型用在图像的生成上。那么像素的取值只依赖于它在空间上的某种近邻。现在比较流行的自回归模型，包括最近刚刚提出的像素CNN或者像素RNN，它们可以用于图像或者视频的生成。



