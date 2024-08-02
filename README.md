# How to eat Pytorch in 20 days ?🔥🔥


## 一，本书📖面向读者 👼


**本书假定读者有一定的机器学习和深度学习基础，使用过Keras或TensorFlow或Pytorch搭建训练过简单的模型。**


🔥🔥**号外号外，《20天吃掉那只Pytorch》视频版本登录BiliBili啦，吃货本货倾情掌勺，只为最纯正的乡土味道，欢迎新老朋友前来品尝** 🍉🍉！

https://www.bilibili.com/video/BV1Ua411P7oe
    

## 二，本书写作风格 🍉


**本书是一本对人类用户极其友善的Pytorch入门工具书，Don't let me think是本书的最高追求。**

本书主要是在参考Pytorch官方文档和函数doc文档基础上整理写成的。

尽管Pytorch官方文档已经相当简明清晰，但本书在篇章结构和范例选取上做了大量的优化，在用户友好度方面更胜一筹。

本书按照内容难易程度、读者检索习惯和Pytorch自身的层次结构设计内容，循序渐进，层次清晰，方便按照功能查找相应范例。

本书在范例设计上尽可能简约化和结构化，增强范例易读性和通用性，大部分代码片段在实践中可即取即用。

**如果说通过学习Pytorch官方文档掌握Pytorch的难度大概是5，那么通过本书学习掌握Pytorch的难度应该大概是2.**

仅以下图对比Pytorch官方文档与本书《20天吃掉那只Pytorch》的差异。



![](https://tva1.sinaimg.cn/large/e6c9d24egy1h536b2yro2j20k00b9myd.jpg)

```python

```

## 三，本书学习方案 ⏰

**1，学习计划**

本书是作者利用工作之余大概3个月写成的，大部分读者应该在20天可以完全学会。

预计每天花费的学习时间在30分钟到2个小时之间。

当然，本书也非常适合作为Pytorch的工具手册在工程落地时作为范例库参考。

**点击学习内容蓝色标题即可进入该章节。**


|日期 | 学习内容                                                       | 内容难度   | 预计学习时间 | 更新状态|
|----:|:--------------------------------------------------------------|-----------:|----------:|-----:|
|&nbsp;|[**一、Pytorch的建模流程**](./一、Pytorch的建模流程.ipynb)    |⭐️   |   0hour   |✅    | 
|day1 | [1-1,结构化数据建模流程范例](./1-1,结构化数据建模流程范例.ipynb)    | ⭐️⭐️⭐️ |   1hour    |✅    |
|day2 | [1-2,图片数据建模流程范例](./1-2,图片数据建模流程范例.ipynb)    | ⭐️⭐️⭐️⭐️  |   2hour    | ✅   |
|day3 | [1-3,文本数据建模流程范例](./1-3,文本数据建模流程范例.ipynb)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    | ✅  |
|day4 | [1-4,时间序列数据建模流程范例](./1-4,时间序列数据建模流程范例.ipynb)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    | ✅   |
|&nbsp; |[**二、Pytorch的核心概念**](./二、Pytorch的核心概念.ipynb)  | ⭐️  |  0hour |✅  |
|day5 |  [2-1,张量数据结构](./2-1,张量数据结构.ipynb)  | ⭐️⭐️⭐️⭐️   |   1hour    | ✅   |
|day6 |  [2-2,自动微分机制](./2-2,自动微分机制.ipynb)  | ⭐️⭐️⭐️   |   1hour    | ✅  |
|day7 |  [2-3,动态计算图](./2-3,动态计算图.ipynb)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    | ✅   |
|&nbsp; |[**三、Pytorch的层次结构**](./三、Pytorch的层次结构.ipynb) |   ⭐️  |  0hour   | ✅  |
|day8 |  [3-1,低阶API示范](./3-1,低阶API示范.ipynb)   | ⭐️⭐️⭐️⭐️   |   1hour    | ✅  |
|day9 |  [3-2,中阶API示范](./3-2,中阶API示范.ipynb)   | ⭐️⭐️⭐️   |  1hour    |✅  |
|day10 | [3-3,高阶API示范](./3-3,高阶API示范.ipynb)  | ⭐️⭐️⭐️  |   1hour    |✅ |
|&nbsp; |[**四、Pytorch的低阶API**](./四、Pytorch的低阶API.ipynb) |⭐️    | 0hour| ✅ |
|day11|  [4-1,张量的结构操作](./4-1,张量的结构操作.ipynb)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    | ✅ |
|day12|  [4-2,张量的数学运算](./4-2,张量的数学运算.ipynb)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅ |
|day13|  [4-3,nn.functional和nn.Module](./4-3,nn.functional和nn.Module.ipynb)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅ |
|&nbsp; |[**五、Pytorch的中阶API**](./五、Pytorch的中阶API.ipynb) |  ⭐️  | 0hour|✅ |
|day14|  [5-1,Dataset和DataLoader](./5-1,Dataset和DataLoader.ipynb)   | ⭐️⭐️⭐️⭐️   |   1hour    | ✅   |
|day15|  [5-2,模型层](./5-2,模型层.ipynb)  | ⭐️⭐️⭐️⭐️⭐️ |   2hour    |✅  |
|day16|  [5-3,损失函数](./5-3,损失函数.ipynb)    | ⭐️⭐️⭐️⭐️   |   1hour    |✅   |
|day17|  [5-4,TensorBoard可视化](./5-4,TensorBoard可视化.ipynb)    | ⭐️⭐️⭐️   |   1hour    | ✅   |
|&nbsp; |[**六、Pytorch的高阶API**](./六、Pytorch的高阶API.ipynb)|    ⭐️ | 0hour|✅  |
|day18|  [6-1,构建模型的3种方法](./6-1,构建模型的3种方法.ipynb)   | ⭐️⭐️    |   0.5hour    |✅   |
|day19|  [6-2,训练模型的3种方法](./6-2,训练模型的3种方法.ipynb)  | ⭐️⭐️⭐️   |   1hour    | ✅  |
|day20|  [6-3,使用GPU训练模型](./6-3,使用GPU训练模型.ipynb)    | ⭐️⭐️⭐️⭐️ |   1hour    | ✅  |
| * |  [后记：我的产品观](https://mp.weixin.qq.com/s/WXUJ0D2iAIWASlkpv60FLA)    | ⭐️   |   0hour    | ✅  |

```python

```

**2，学习环境**

本书全部源码在jupyter中编写测试通过，建议通过git克隆到本地，并在jupyter中交互式运行学习。

step1: 克隆本书源码到本地,使用码云镜像仓库国内下载速度更快
```
git clone https://gitee.com/Python_Ai_Road/eat_pytorch_in_20_days
```

step2: 公众号 **算法美食屋** 回复关键词：**pytorch**， 获取本项目所用数据集汇总压缩包 eat_pytorch_datasets.zip百度云盘下载链接，下载解压并移动到eat_pytorch_in_20_days路径下，约160M。




```python
import torch 
from torch import nn

print("torch version:", torch.__version__)

a = torch.tensor([[2,1]])
b = torch.tensor([[-1,2]])
c = a@b.t()
print("[[2,1]]@[[-1],[2]] =", c.item())

```

```
torch version: 2.0.1
[[2,1]]@[[-1],[2]] = 0
```



## 四，项目更新记录


### 1，2022-08🎈🎈更新 **pytorch与广告推荐**章节

适合对广告推荐领域感兴趣，且需要进阶的同学😋😋


|日期 | 学习内容                                                       | 内容难度   | 预计学习时间 | 更新状态|
|----:|:--------------------------------------------------------------|-----------:|----------:|-----:|
|&nbsp; |[**七、Pytorch与广告推荐**](./七、Pytorch与广告推荐.ipynb)|    ⭐️ | 0hour|✅  |
|day1|  [7-1,推荐算法业务](./7-1,推荐算法业务.ipynb)   | ⭐️⭐️⭐️    |   0.5hour    |✅   |
|day2|  [7-2,广告算法业务](./7-2,广告算法业务.ipynb)  | ⭐️⭐️⭐️   |   0.5hour    | ✅  |
|day3|  [7-3,FM模型](./7-3,FM模型.ipynb)    | ⭐️⭐️⭐️   |   1hour    | ✅  |
|day4|  [7-4,DeepFM模型](./7-4,DeepFM模型.ipynb)    | ⭐️⭐️⭐️⭐️    |   1hour    | ✅  |
|day5|  [7-5,FiBiNET模型](./7-5,FiBiNET模型.ipynb)    | ⭐️⭐️⭐️⭐️    |   2hour    | ✅  |
|day6|  [7-6,DeepCross模型](./7-6,DeepCross模型.ipynb)    | ⭐️⭐️⭐️⭐️⭐️    |   2hour    | ✅  |
|day7|  [7-7,DIN网络](./7-7,DIN网络.ipynb)    | ⭐️⭐️⭐️⭐️⭐️    |   2hour    | ✅  |
|day8|  [7-8,DIEN网络](./7-8,DIEN网络.ipynb)    | ⭐️⭐️⭐️⭐️⭐️    |   2hour    | ✅  |



### 2，2023-03🎈🎈更新 彩蛋章节

介绍一些与pytorch相关的周边工具


|日期 | 学习内容                                                       | 内容难度   | 预计学习时间 | 更新状态|
|----:|:--------------------------------------------------------------|-----------:|----------:|-----:|
|&nbsp; |[**彩蛋：Pytorch周边工具**](./彩蛋：Pytorch周边工具.ipynb)|    ⭐️ | 0hour|✅  |
|day1|  [A-1, Kaggle免费GPU使用攻略](./A-1,Kaggle免费GPU使用攻略.ipynb)   | ⭐️⭐️⭐️    |   1hour    |✅   |
|day2|  [A-2, Streamlit构建机器学习应用](./A-2,Streamlit构建机器学习应用.ipynb)  | ⭐️⭐️⭐️   |  1hour    | ✅  |
|day3| [A-3, 使用Mac M1芯片加速pytorch](./A-3,使用MacM1芯片加速pytorch.ipynb) | ⭐️⭐️⭐️   |  1hour    | ✅  |
|day4| [A-4, optuna可视化调参魔法指南](./A-4,optuna可视化调参魔法指南.ipynb) | ⭐️⭐️⭐️⭐️   |  1hour    | ✅  |
|day5| [A-5, gradio让你的机器学习模型性感起来](./A-5,Gradio让你的机器学习模型性感起来.ipynb) | ⭐️⭐️⭐️⭐️   |  1hour    | ✅  |
|day6| [A-6, wandb模型可视化分析](./A-6,30分钟吃掉wandb可视化模型分析.ipynb) | ⭐️⭐️⭐️⭐   |  0.5hour    | ✅  |
|day7| [A-7, wandb模型可视化自动调参](./A-7,30分钟吃掉wandb可视化自动调参.ipynb) | ⭐️⭐️⭐️⭐  |  1hour    | ✅  |




### 3， 2023-07🎈🎈更新pytorch模型训练工具库torchkeras

相关章节代码进行了对应优化调整。


|功能| 稳定支持起始版本 | 依赖或借鉴库 |
|:----|:-------------------:|:--------------|
|✅ 训练进度条 | 3.0.0   | 依赖tqdm,借鉴keras|
|✅ 训练评估指标  | 3.0.0   | 借鉴pytorch_lightning |
|✅ notebook中训练自带可视化 |  3.8.0  |借鉴fastai |
|✅ early stopping | 3.0.0   | 借鉴keras |
|✅ gpu training | 3.0.0    |依赖accelerate|
|✅ multi-gpus training(ddp) |   3.6.0 | 依赖accelerate|
|✅ fp16/bf16 training|   3.6.0  | 依赖accelerate|
|✅ tensorboard callback |   3.7.0  |依赖tensorboard |
|✅ wandb callback |  3.7.0 |依赖wandb |


详情参考项目链接：：https://github.com/lyhue1991/torchkeras 



```python

```

## 五，鼓励和联系作者 🎈🎈


**如果本书对你有所帮助，想鼓励一下作者，记得给本项目加一颗星星star⭐️，并分享给你的朋友们喔😊!** 

如果对本书内容理解上有一些疑问或者建议，可以在公众号"算法美食屋"后台回复关键字：**加群**，加入读者交流群和大家讨论。

![算法美食屋logo.png](https://tva1.sinaimg.cn/large/e6c9d24egy1h41m2zugguj20k00b9q46.jpg)

```python

```
