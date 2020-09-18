# How to eat Pytorch in 20 days ?🔥🔥

**《20天吃掉那只Pytorch》**
* 🚀 github项目地址: https://github.com/lyhue1991/eat_pytorch_in_20_days
* 🐳 和鲸专栏地址: https://www.kesci.com/home/column/5f2ac5d8af3980002cb1bc08 【代码可直接fork后云端运行，无需配置环境】

**《30天吃掉那只TensorFlow2》**
* 🚀 github项目地址: https://github.com/lyhue1991/eat_tensorflow2_in_30_days
* 🐳 和鲸专栏地址: https://www.kesci.com/home/column/5d8ef3c3037db3002d3aa3a0 【代码可直接fork后云端运行，无需配置环境】

<details><summary>Chinese</summary>

### 一， Pytorch🔥  or TensorFlow2 🍎 

先说结论:

**如果是工程师，应该优先选TensorFlow2.**

**如果是学生或者研究人员，应该优先选择Pytorch.**

**如果时间足够，最好TensorFlow2和Pytorch都要学习掌握。**


理由如下：

* 1，**在工业界最重要的是模型落地，目前国内的大部分互联网企业只支持TensorFlow模型的在线部署，不支持Pytorch。** 并且工业界更加注重的是模型的高可用性，许多时候使用的都是成熟的模型架构，调试需求并不大。


* 2，**研究人员最重要的是快速迭代发表文章，需要尝试一些较新的模型架构。而Pytorch在易用性上相比TensorFlow2有一些优势，更加方便调试。** 并且在2019年以来在学术界占领了大半壁江山，能够找到的相应最新研究成果更多。


* 3，TensorFlow2和Pytorch实际上整体风格已经非常相似了，学会了其中一个，学习另外一个将比较容易。两种框架都掌握的话，能够参考的开源模型案例更多，并且可以方便地在两种框架之间切换。


本书的TensorFlow镜像教程：

#### 🍊《30天吃掉那只TensorFlow2》：https://github.com/lyhue1991/eat_tensorflow2_in_30_days 

```python

```

### 二，本书📖面向读者 👼


**本书假定读者有一定的机器学习和深度学习基础，使用过Keras或TensorFlow或Pytorch搭建训练过简单的模型。**

**对于没有任何机器学习和深度学习基础的同学，建议在学习本书时同步参考阅读《Python深度学习》一书的第一部分"深度学习基础"内容。**

《Python深度学习》这本书是Keras之父Francois Chollet所著，该书假定读者无任何机器学习知识，以Keras为工具，

使用丰富的范例示范深度学习的最佳实践，该书通俗易懂，**全书没有一个数学公式，注重培养读者的深度学习直觉。**。

《Python深度学习》一书的第一部分的4个章节内容如下，预计读者可以在20小时之内学完。

* 1，什么是深度学习

* 2，神经网络的数学基础

* 3，神经网络入门

* 4，机器学习基础


```python

```

### 三，本书写作风格 🍉


**本书是一本对人类用户极其友善的Pytorch入门工具书，Don't let me think是本书的最高追求。**

本书主要是在参考Pytorch官方文档和函数doc文档基础上整理写成的。

尽管Pytorch官方文档已经相当简明清晰，但本书在篇章结构和范例选取上做了大量的优化，在用户友好度方面更胜一筹。

本书按照内容难易程度、读者检索习惯和Pytorch自身的层次结构设计内容，循序渐进，层次清晰，方便按照功能查找相应范例。

本书在范例设计上尽可能简约化和结构化，增强范例易读性和通用性，大部分代码片段在实践中可即取即用。

**如果说通过学习Pytorch官方文档掌握Pytorch的难度大概是5，那么通过本书学习掌握Pytorch的难度应该大概是2.**

仅以下图对比Pytorch官方文档与本书《20天吃掉那只Pytorch》的差异。



![](./data/Pytorch官方vs吃掉Pytorch.png)

```python

```

### 四，本书学习方案 ⏰

**1，学习计划**

本书是作者利用工作之余大概3个月写成的，大部分读者应该在20天可以完全学会。

预计每天花费的学习时间在30分钟到2个小时之间。

当然，本书也非常适合作为Pytorch的工具手册在工程落地时作为范例库参考。

**点击学习内容蓝色标题即可进入该章节。**


|日期 | 学习内容                                                       | 内容难度   | 预计学习时间 | 更新状态|
|----:|:--------------------------------------------------------------|-----------:|----------:|-----:|
|&nbsp;|[**一、Pytorch的建模流程**](./一、Pytorch的建模流程.md)    |⭐️   |   0hour   |✅    |
|day1 | [1-1,结构化数据建模流程范例](./1-1,结构化数据建模流程范例.md)    | ⭐️⭐️⭐️ |   1hour    |✅    |
|day2 | [1-2,图片数据建模流程范例](./1-2,图片数据建模流程范例.md)    | ⭐️⭐️⭐️⭐️  |   2hour    | ✅   |
|day3 | [1-3,文本数据建模流程范例](./1-3,文本数据建模流程范例.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    | ✅  |
|day4 | [1-4,时间序列数据建模流程范例](./1-4,时间序列数据建模流程范例.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    | ✅   |
|&nbsp; |[**二、Pytorch的核心概念**](./二、Pytorch的核心概念.md)  | ⭐️  |  0hour |✅  |
|day5 |  [2-1,张量数据结构](./2-1,张量数据结构.md)  | ⭐️⭐️⭐️⭐️   |   1hour    | ✅   |
|day6 |  [2-2,自动微分机制](./2-2,自动微分机制.md)  | ⭐️⭐️⭐️   |   1hour    | ✅  |
|day7 |  [2-3,动态计算图](./2-3,动态计算图.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    | ✅   |
|&nbsp; |[**三、Pytorch的层次结构**](./三、Pytorch的层次结构.md) |   ⭐️  |  0hour   | ✅  |
|day8 |  [3-1,低阶API示范](./3-1,低阶API示范.md)   | ⭐️⭐️⭐️⭐️   |   1hour    | ✅  |
|day9 |  [3-2,中阶API示范](./3-2,中阶API示范.md)   | ⭐️⭐️⭐️   |  1hour    |✅  |
|day10 | [3-3,高阶API示范](./3-3,高阶API示范.md)  | ⭐️⭐️⭐️  |   1hour    |✅ |
|&nbsp; |[**四、Pytorch的低阶API**](./四、Pytorch的低阶API.md) |⭐️    | 0hour| ✅ |
|day11|  [4-1,张量的结构操作](./4-1,张量的结构操作.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    | ✅ |
|day12|  [4-2,张量的数学运算](./4-2,张量的数学运算.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅ |
|day13|  [4-3,nn.functional和nn.Module](./4-3,nn.functional和nn.Module.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅ |
|&nbsp; |[**五、Pytorch的中阶API**](./五、Pytorch的中阶API.md) |  ⭐️  | 0hour|✅ |
|day14|  [5-1,Dataset和DataLoader](./5-1,Dataset和DataLoader.md)   | ⭐️⭐️⭐️⭐️⭐️   |   2hour    | ✅   |
|day15|  [5-2,模型层](./5-3,模型层.md)  | ⭐️⭐️⭐️   |   1hour    |✅  |
|day16|  [5-3,损失函数](./5-4,损失函数.md)    | ⭐️⭐️⭐️   |   1hour    |✅   |
|day17|  [5-4,TensorBoard可视化](./5-4,TensorBoard可视化.md)    | ⭐️⭐️⭐️   |   1hour    | ✅   |
|&nbsp; |[**六、Pytorch的高阶API**](./六、Pytorch的高阶API.md)|    ⭐️ | 0hour|✅  |
|day18|  [6-1,构建模型的3种方法](./6-1,构建模型的3种方法.md)   | ⭐️⭐️⭐️⭐️    |   1hour    |✅   |
|day19|  [6-2,训练模型的3种方法](./6-2,训练模型的3种方法.md)  | ⭐️⭐️⭐️⭐️   |   1hour    | ✅  |
|day20|  [6-3,使用GPU训练模型](./6-3,使用GPU训练模型.md)    | ⭐️⭐️⭐️⭐️    |   1hour    | ✅  |



```python

```

**2，学习环境**


本书全部源码在jupyter中编写测试通过，建议通过git克隆到本地，并在jupyter中交互式运行学习。

为了直接能够在jupyter中打开markdown文件，建议安装jupytext，将markdown转换成ipynb文件。

```python
#克隆本书源码到本地,使用码云镜像仓库国内下载速度更快
#!git clone https://gitee.com/Python_Ai_Road/eat_pytorch_in_20_days

#建议在jupyter notebook 上安装jupytext，以便能够将本书各章节markdown文件视作ipynb文件运行
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U jupytext
    
#建议在jupyter notebook 上安装最新版本pytorch 测试本书中的代码
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  -U torch torchvision torchtext torchkeras 
```

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
torch version: 1.5.0
[[2,1]]@[[-1],[2]] = 0
```

```python

```

### 五，鼓励和联系作者 🎈🎈


**如果本书对你有所帮助，想鼓励一下作者，记得给本项目加一颗星星star⭐️，并分享给你的朋友们喔😊!** 

如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"Python与算法之美"下留言。作者时间和精力有限，会酌情予以回复。

也可以在公众号后台回复关键字：**加群**，加入读者交流群和大家讨论。

![image.png](./data/Python与算法之美logo.jpg)

```python

```
</details>

<details><summary>English</summary>

** "Eat Pytorch in 20 Days" **
* 🚀 github project address: https://github.com/lyhue1991/eat_pytorch_in_20_days
* 🐳 Column address: https://www.kesci.com/home/column/5f2ac5d8af3980002cb1bc08 【Code can be run directly in the cloud after fork, no need to configure the environment】

**"Eat TensorFlow 2 in 30 Days"**
* 🚀 github project address: https://github.com/lyhue1991/eat_tensorflow2_in_30_days
* 🐳 Column address: https://www.kesci.com/home/column/5d8ef3c3037db3002d3aa3a0 【The code can be run directly in the cloud after fork, no need to configure the environment】

### 1. Pytorch🔥  or TensorFlow2 🍎 

Conclusion first:

**If you are an engineer, TensorFlow2 should be preferred.**

**If you are a student or researcher, Pytorch should be preferred.**

**If there is enough time, it is best to learn and master both TensorFlow2 and Pytorch.**

Why to master both?

* 1. **The most important thing in the industry is the production of models. At present, most domestic Internet companies only support the online deployment of TensorFlow models, not Pytorch.** And the industry pays more attention to the high availability of the model. Many times, the mature model architectures are used, and the need for debugging is not really large.

* 2. **In research, the most important thing is to publish articles quickly, and they need to try some newer model architectures. Pytorch has some advantages over TensorFlow2 in terms of ease of use and is more convenient for debugging.** Pytorch has occupied more than half of the academic world since 2019 with more cutting-edge research results.

* 3. TensorFlow2 and Pytorch are actually very similar in overall style. After learning one, it will be easier to learn the other. Mastering both frameworks provides you oppurtunities to contribute to more open source model cases.

For mastering Tensorflow:

#### 🍊 "Eat TensorFlow2 in 30 days" ： https: //github.com/lyhue1991/eat_tensorflow2_in_30_days

```python

```

### 2. What Should You Know Before Reading This Book? 👼

This book assumes that the reader has a certain foundation of machine learning and deep learning, and has used Keras or TensorFlow or Pytorch to build and train simple models.

For students who do not have any machine learning and deep learning foundations, it is recommended to read the first part of the book **"Deep Learning with Python"** when studying this book.

The book **"Deep Learning with Python"** is written by Francois Chollet, the father of Keras. The book assumes that the reader has no machine learning knowledge and uses Keras as a tool. It uses many examples to demonstrate the best practices of deep learning. The book is easy to understand as there is no mathematical formula in the book. The book mainly focuses on cultivating readers' deep learning intuition.

The contents of the 4 chapters of the first part of the book "Deep Learning for python" are as follows:

1. What is deep learning 
2. The mathematical building blocks of neural networks 
3. Getting started with Neural Networks 
4. Fundamentals of Machine learning

```python

```

### 3. Writing style of this book 🍉


**This book is a Pytorch introductory tool that is extremely friendly to human users. "Don't let the readers think" is the highest pursuit of this book.**

This book is mainly organized and written on the basis of referring to Pytorch official documentation together with its functions.

Although the official Pytorch documentation is quite concise and clear, this book has made a lot of optimizations in the chapter structure and selection of examples, which is more user-friendly.

This book is designed in accordance with the difficulty of the content, the reader's search habits and Pytorch's own hierarchical structure. The content is designed step by step, with clear levels, and it is convenient to find corresponding examples according to functions.

This book is as simple and structured as possible in the design of examples to enhance the legibility and versatility of examples. Most of the code snippets are ready to use in practice.

**If the difficulty of mastering Pytorch by learning the official Pytorch documentation is about 5, then the difficulty of learning to master Pytorch through this book should be about 2.**

```python

```

### 4. How to use this Book? ⏰


**1. Study Plan**

Number of days required to eat this book: This book was written by the author about 3 months after work, and most readers should be able to learn it in **20 days**.

How many hours a day should you spend: It is estimated that the study time spent every day is between 30 minutes and 2 hours.

Note: This book is also very suitable as a reference for Pytorch's tool manual when the project is implemented.

**Click the blue title of the learning content to enter the chapter.**

|Date | Contents                                                       | Difficulty   | Est. Time | Update Status|
|----:|:--------------------------------------------------------------|-----------:|----------:|-----:|
|&nbsp;|[**1. Pytorch's modeling process**](./一、Pytorch的建模流程.md)    |⭐️   |   0hour   |✅    |
|day1 | [1-1. Example of structured data modeling process](./1-1,结构化数据建模流程范例.md)    | ⭐️⭐️⭐️ |   1hour    |✅    |
|day2 | [1-2. Example of image data modeling process](./1-2,图片数据建模流程范例.md)    | ⭐️⭐️⭐️⭐️  |   2hour    | ✅   |
|day3 | [1-3. Example of text data modeling process](./1-3,文本数据建模流程范例.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    | ✅  |
|day4 | [1-4. Example of time series data modeling process](./1-4,时间序列数据建模流程范例.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    | ✅   |
|&nbsp; |[**2. The core concept of Pytorch**](./二、Pytorch的核心概念.md)  | ⭐️  |  0hour |✅  |
|day5 |  [2-1. Tensor data structure](./2-1,张量数据结构.md)  | ⭐️⭐️⭐️⭐️   |   1hour    | ✅   |
|day6 |  [2-2. Automatic differentiation mechanism](./2-2,自动微分机制.md)  | ⭐️⭐️⭐️   |   1hour    | ✅  |
|day7 |  [2-3. Dynamic calculation diagram](./2-3,动态计算图.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    | ✅   |
|&nbsp; |[**3. The hierarchy of Pytorch**](./三、Pytorch的层次结构.md) |   ⭐️  |  0hour   | ✅  |
|day8 |  [3-1. Low-level API demonstration](./3-1,低阶API示范.md)   | ⭐️⭐️⭐️⭐️   |   1hour    | ✅  |
|day9 |  [3-2. Intermediate API demonstration](./3-2,中阶API示范.md)   | ⭐️⭐️⭐️   |  1hour    |✅  |
|day10 | [3-3. High-level API demonstration](./3-3,高阶API示范.md)  | ⭐️⭐️⭐️  |   1hour    |✅ |
|&nbsp; |[**4. Pytorch's low-level API**](./四、Pytorch的低阶API.md) |⭐️    | 0hour| ✅ |
|day11|  [4-1. Tensor structure operation](./4-1,张量的结构操作.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    | ✅ |
|day12|  [4-2. Mathematical operations of tensors](./4-2,张量的数学运算.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅ |
|day13|  [4-3. nn.functional and nn.Module](./4-3,nn.functional和nn.Module.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅ |
|&nbsp; |[**5. Pytorch's intermediate-level API**](./五、Pytorch的中阶API.md) |  ⭐️  | 0hour|✅ |
|day14|  [5-1. Dataset and DataLoader](./5-1,Dataset和DataLoader.md)   | ⭐️⭐️⭐️⭐️⭐️   |   2hour    | ✅   |
|day15|  [5-2. Model layer](./5-3,模型层.md)  | ⭐️⭐️⭐️   |   1hour    |✅  |
|day16|  [5-3. Loss function](./5-4,损失函数.md)    | ⭐️⭐️⭐️   |   1hour    |✅   |
|day17|  [5-4. TensorBoard TensorBoard visualization](./5-4,TensorBoard可视化.md)    | ⭐️⭐️⭐️   |   1hour    | ✅   |
|&nbsp; |[**6. Pytorch's high-level API**](./六、Pytorch的高阶API.md)|    ⭐️ | 0hour|✅  |
|day18|  [6-1. 3 ways to build a model](./6-1,构建模型的3种方法.md)   | ⭐️⭐️⭐️⭐️    |   1hour    |✅   |
|day19|  [6-2. 3 ways to train a model](./6-2,训练模型的3种方法.md)  | ⭐️⭐️⭐️⭐️   |   1hour    | ✅  |
|day20|  [6-3. Use GPU to train model](./6-3,使用GPU训练模型.md)    | ⭐️⭐️⭐️⭐️    |   1hour    | ✅  |

**2. Learning environment**

All the source code of this book has been written and tested in jupyter. It is recommended to clone to the local through git and run and learn interactively in jupyter.

In order to directly open the markdown file in jupyter, it is recommended to install jupytext and convert the markdown to an ipynb file.

```python
#clone the source code of this book to local, use the code cloud mirror warehouse to download faster in china

#!git clone https://gitee.com/Python_Ai_Road/eat_pytorch_in_20_days

#it is recommended to install jupytext on jupyter notebook so that the markdown files of each chapter of this book can be run as ipynb files
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U jupytext
    
#it is recommended to install the latest version of pytorch on jupyter notebook to test the code in this book
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  -U torch torchvision torchtext torchkeras 
```

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
torch version: 1.5.0
[[2,1]]@[[-1],[2]] = 0
```
```python

```

### 5. Contact and support the author 🎈🎈


**If this book is helpful to you and want to encourage the author, remember to add a star⭐️ to this project and share it with your friends😊!** 

If you need to further communicate with the author on the understanding of the content of this book, please leave a message under the public account "The Beauty of Python and Algorithms". The author has limited time and energy and will respond as appropriate.

You can also reply to keywords in the background of the official account: add group, join the reader exchange group and discuss with you.

![image.png](./data/Python与算法之美logo.jpg)

```python

```

</details>
