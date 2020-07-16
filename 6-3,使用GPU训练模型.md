# 6-3,使用GPU训练模型


深度学习的训练过程常常非常耗时，一个模型训练几个小时是家常便饭，训练几天也是常有的事情，有时候甚至要训练几十天。

训练过程的耗时主要来自于两个部分，一部分来自数据准备，另一部分来自参数迭代。

当数据准备过程还是模型训练时间的主要瓶颈时，我们可以使用更多进程来准备数据。

当参数迭代过程成为训练时间的主要瓶颈时，我们通常的方法是应用GPU来进行加速。

<!-- #region -->
Pytorch中使用GPU加速模型非常简单，只要将模型和数据移动到GPU上。核心代码只有以下几行。

```python
# 定义模型
... 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device) # 移动模型到cuda

# 训练模型
...

features = features.to(device) # 移动数据到cuda
labels = labels.to(device) # 或者  labels = labels.cuda() if torch.cuda.is_available() else labels
...
```

如果要使用多个GPU训练模型，也非常简单。只需要在将模型设置为数据并行风格模型。
则模型移动到GPU上之后，会在每一个GPU上拷贝一个副本，并把数据平分到各个GPU上进行训练。核心代码如下。

```python
# 定义模型
... 

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model) # 包装为并行风格模型

# 训练模型
...
features = features.to(device) # 移动数据到cuda
labels = labels.to(device) # 或者 labels = labels.cuda() if torch.cuda.is_available() else labels
...
```
<!-- #endregion -->

**以下是一些和GPU有关的基本操作汇总** 


在Colab笔记本中：修改->笔记本设置->硬件加速器 中选择 GPU

注：以下代码只能在Colab 上才能正确执行。

可点击如下链接，直接在colab中运行范例代码。

《torch使用gpu训练模型》

https://colab.research.google.com/drive/1FDmi44-U3TFRCt9MwGn4HIj2SaaWIjHu?usp=sharing

```python
import torch 
from torch import nn 
```

```python
# 1，查看gpu信息
if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)

gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)

```

```
if_cuda= True
gpu_count= 1
```

```python
# 2，将张量在gpu和cpu间移动
tensor = torch.rand((100,100))
tensor_gpu = tensor.to("cuda:0") # 或者 tensor_gpu = tensor.cuda()
print(tensor_gpu.device)
print(tensor_gpu.is_cuda)

tensor_cpu = tensor_gpu.to("cpu") # 或者 tensor_cpu = tensor_gpu.cpu() 
print(tensor_cpu.device)

```

```
cuda:0
True
cpu
```

```python
# 3，将模型中的全部张量移动到gpu上
net = nn.Linear(2,1)
print(next(net.parameters()).is_cuda)
net.to("cuda:0") # 将模型中的全部参数张量依次到GPU上，注意，无需重新赋值为 net = net.to("cuda:0")
print(next(net.parameters()).is_cuda)
print(next(net.parameters()).device)
```

```
False
True
cuda:0
```

```python
# 4，创建支持多个gpu数据并行的模型
linear = nn.Linear(2,1)
print(next(linear.parameters()).device)

model = nn.DataParallel(linear)
print(model.device_ids)
print(next(model.module.parameters()).device) 

#注意保存参数时要指定保存model.module的参数
torch.save(model.module.state_dict(), "./data/model_parameter.pkl") 

linear = nn.Linear(2,1)
linear.load_state_dict(torch.load("./data/model_parameter.pkl")) 

```

```
cpu
[0]
cuda:0
```

```python
# 5，清空cuda缓存

# 该方法在cuda超内存时十分有用
torch.cuda.empty_cache()

```

```python

```

### 一，矩阵乘法范例


下面分别使用CPU和GPU作一个矩阵乘法，并比较其计算效率。

```python
import time
import torch 
from torch import nn
```

```python
# 使用cpu
a = torch.rand((10000,200))
b = torch.rand((200,10000))
tic = time.time()
c = torch.matmul(a,b)
toc = time.time()

print(toc-tic)
print(a.device)
print(b.device)
```

```
0.6454010009765625
cpu
cpu
```

```python
# 使用gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.rand((10000,200),device = device) #可以指定在GPU上创建张量
b = torch.rand((200,10000)) #也可以在CPU上创建张量后移动到GPU上
b = b.to(device) #或者 b = b.cuda() if torch.cuda.is_available() else b 
tic = time.time()
c = torch.matmul(a,b)
toc = time.time()
print(toc-tic)
print(a.device)
print(b.device)

```

```
0.014541149139404297
cuda:0
cuda:0
```


### 二，线性回归范例



下面对比使用CPU和GPU训练一个线性回归模型的效率


**1，使用CPU**

```python
# 准备数据
n = 1000000 #样本数量

X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0,-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0.t() + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动
```

```python
# 定义模型
class LinearRegression(nn.Module): 
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))
    #正向传播
    def forward(self,x): 
        return x@self.w.t() + self.b
        
linear = LinearRegression() 

```

```python
# 训练模型
optimizer = torch.optim.Adam(linear.parameters(),lr = 0.1)
loss_func = nn.MSELoss()

def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        optimizer.zero_grad()
        Y_pred = linear(X) 
        loss = loss_func(Y_pred,Y)
        loss.backward() 
        optimizer.step()
        if epoch%50==0:
            print({"epoch":epoch,"loss":loss.item()})
    toc = time.time()
    print("time used:",toc-tic)

train(500)
```

```
{'epoch': 0, 'loss': 3.996487855911255}
{'epoch': 50, 'loss': 3.9969770908355713}
{'epoch': 100, 'loss': 3.9964890480041504}
{'epoch': 150, 'loss': 3.996488332748413}
{'epoch': 200, 'loss': 3.996488094329834}
{'epoch': 250, 'loss': 3.996488332748413}
{'epoch': 300, 'loss': 3.996488332748413}
{'epoch': 350, 'loss': 3.996488094329834}
{'epoch': 400, 'loss': 3.996488332748413}
{'epoch': 450, 'loss': 3.996488094329834}
time used: 5.4090576171875
```




**2，使用GPU**

```python
# 准备数据
n = 1000000 #样本数量

X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0,-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0.t() + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动

# 移动到GPU上
print("torch.cuda.is_available() = ",torch.cuda.is_available())
X = X.cuda()
Y = Y.cuda()
print("X.device:",X.device)
print("Y.device:",Y.device)
```

```
torch.cuda.is_available() =  True
X.device: cuda:0
Y.device: cuda:0
```

```python
# 定义模型
class LinearRegression(nn.Module): 
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))
    #正向传播
    def forward(self,x): 
        return x@self.w.t() + self.b
        
linear = LinearRegression() 

# 移动模型到GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
linear.to(device)

#查看模型是否已经移动到GPU上
print("if on cuda:",next(linear.parameters()).is_cuda)

```

```
if on cuda: True
```

```python
# 训练模型
optimizer = torch.optim.Adam(linear.parameters(),lr = 0.1)
loss_func = nn.MSELoss()

def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        optimizer.zero_grad()
        Y_pred = linear(X) 
        loss = loss_func(Y_pred,Y)
        loss.backward() 
        optimizer.step()
        if epoch%50==0:
            print({"epoch":epoch,"loss":loss.item()})
    toc = time.time()
    print("time used:",toc-tic)
    
train(500)
```

```
{'epoch': 0, 'loss': 3.9982845783233643}
{'epoch': 50, 'loss': 3.998818874359131}
{'epoch': 100, 'loss': 3.9982895851135254}
{'epoch': 150, 'loss': 3.9982845783233643}
{'epoch': 200, 'loss': 3.998284339904785}
{'epoch': 250, 'loss': 3.9982845783233643}
{'epoch': 300, 'loss': 3.9982845783233643}
{'epoch': 350, 'loss': 3.9982845783233643}
{'epoch': 400, 'loss': 3.9982845783233643}
{'epoch': 450, 'loss': 3.9982845783233643}
time used: 0.4889392852783203
```

```python

```

### 三，torchkeras使用单GPU范例


下面演示使用torchkeras来应用GPU训练模型的方法。

其对应的CPU训练模型代码参见《6-2,训练模型的3种方法》

本例仅需要在它的基础上增加一行代码，在model.compile时指定 device即可。



**1，准备数据**

```python
!pip install -U torchkeras 
```

```python
import torch 
from torch import nn 

import torchvision 
from torchvision import transforms

import torchkeras 
```

```python
transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="./data/minist/",train=True,download=True,transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./data/minist/",train=False,download=True,transform=transform)

dl_train =  torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid =  torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_valid))
```

```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

#查看部分样本
from matplotlib import pyplot as plt 

plt.figure(figsize=(8,8)) 
for i in range(9):
    img,label = ds_train[i]
    img = torch.squeeze(img)
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([]) 
plt.show()
```

```python

```

**2，定义模型**

```python
class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)]
        )
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

net = CnnModel()
model = torchkeras.Model(net)
model.summary(input_shape=(1,32,32))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 30, 30]             320
         MaxPool2d-2           [-1, 32, 15, 15]               0
            Conv2d-3           [-1, 64, 11, 11]          51,264
         MaxPool2d-4             [-1, 64, 5, 5]               0
         Dropout2d-5             [-1, 64, 5, 5]               0
 AdaptiveMaxPool2d-6             [-1, 64, 1, 1]               0
           Flatten-7                   [-1, 64]               0
            Linear-8                   [-1, 32]           2,080
              ReLU-9                   [-1, 32]               0
           Linear-10                   [-1, 10]             330
================================================================
Total params: 53,994
Trainable params: 53,994
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.003906
Forward/backward pass size (MB): 0.359695
Params size (MB): 0.205971
Estimated Total Size (MB): 0.569572
----------------------------------------------------------------
```


**3，训练模型**

```python
from sklearn.metrics import accuracy_score

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu().numpy(),y_pred_cls.cpu().numpy()) 
    # 注意此处要将数据先移动到cpu上，然后才能转换成numpy数组

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # 注意此处compile时指定了device

dfhistory = model.fit(3,dl_train = dl_train, dl_val=dl_valid, log_step_freq=100) 

```

```
Start Training ...

================================================================================2020-06-27 00:24:29
{'step': 100, 'loss': 1.063, 'accuracy': 0.619}
{'step': 200, 'loss': 0.681, 'accuracy': 0.764}
{'step': 300, 'loss': 0.534, 'accuracy': 0.818}
{'step': 400, 'loss': 0.458, 'accuracy': 0.847}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   1   | 0.412 |  0.863   |  0.128   |    0.961     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-27 00:24:35
{'step': 100, 'loss': 0.147, 'accuracy': 0.956}
{'step': 200, 'loss': 0.156, 'accuracy': 0.954}
{'step': 300, 'loss': 0.156, 'accuracy': 0.954}
{'step': 400, 'loss': 0.157, 'accuracy': 0.955}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   2   | 0.153 |  0.956   |  0.085   |    0.976     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-27 00:24:42
{'step': 100, 'loss': 0.126, 'accuracy': 0.965}
{'step': 200, 'loss': 0.147, 'accuracy': 0.96}
{'step': 300, 'loss': 0.153, 'accuracy': 0.959}
{'step': 400, 'loss': 0.147, 'accuracy': 0.96}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   3   | 0.146 |   0.96   |  0.119   |    0.968     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-27 00:24:48
Finished Training...
```


**4，评估模型**

```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
```

```python
plot_metric(dfhistory,"loss")
```

```python
plot_metric(dfhistory,"accuracy")
```

```python
model.evaluate(dl_valid)
```

```
{'val_accuracy': 0.967068829113924, 'val_loss': 0.11601964030650598}
```


**5，使用模型**

```python
model.predict(dl_valid)[0:10]
```

```
tensor([[ -9.2092,   3.1997,   1.4028,  -2.7135,  -0.7320,  -2.0518, -20.4938,
          14.6774,   1.7616,   5.8549],
        [  2.8509,   4.9781,  18.0946,   0.0928,  -1.6061,  -4.1437,   4.8697,
           3.8811,   4.3869,  -3.5929],
        [-22.5231,  13.6643,   5.0244, -11.0188, -16.8147,  -9.5894,  -6.2556,
         -10.5648, -12.1022, -19.4685],
        [ 23.2670, -12.0711,  -7.3968,  -8.2715,  -1.0915, -12.6050,   8.0444,
         -16.9339,   1.8827,  -0.2497],
        [ -4.1159,   3.2102,   0.4971, -11.8064,  12.1460,  -5.1650,  -6.5918,
           1.0088,   0.8362,   2.5132],
        [-26.1764,  15.6251,   6.1191, -12.2424, -13.9725, -10.0540,  -7.8669,
          -5.9602, -11.1944, -18.7890],
        [ -5.0602,   3.3779,  -0.6647,  -8.5185,  10.0320,  -5.5107,  -6.9579,
           2.3811,   0.2542,   3.2860],
        [  4.1017,  -0.4282,   7.2220,   3.3700,  -3.6813,   1.1576,  -1.8479,
           0.7450,   3.9768,   6.2640],
        [  1.9689,  -0.3960,   7.4414, -10.4789,   2.7066,   1.7482,   5.7971,
          -4.5808,   3.0911,  -5.1971],
        [ -2.9680,  -1.2369,  -0.0829,  -1.8577,   1.9380,  -0.8374,  -8.2207,
           3.5060,   3.8735,  13.6762]], device='cuda:0')
```


**6，保存模型**

```python
# save the model parameters
torch.save(model.state_dict(), "model_parameter.pkl")

model_clone = torchkeras.Model(CnnModel())
model_clone.load_state_dict(torch.load("model_parameter.pkl"))

model_clone.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # 注意此处compile时指定了device

model_clone.evaluate(dl_valid)
```

```
{'val_accuracy': 0.967068829113924, 'val_loss': 0.11601964030650598}
```


### 四，torchkeras使用多GPU范例


注：以下范例需要在有多个GPU的机器上跑。如果在单GPU的机器上跑，也能跑通，但是实际上使用的是单个GPU。


**1，准备数据**

```python
import torch 
from torch import nn 

import torchvision 
from torchvision import transforms

import torchkeras 
```

```python
transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="./data/minist/",train=True,download=True,transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./data/minist/",train=False,download=True,transform=transform)

dl_train =  torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid =  torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_valid))
```

**2，定义模型**

```python
class CnnModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)]
        )
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)  
        return x

net = nn.DataParallel(CnnModule())  #Attention this line!!!
model = torchkeras.Model(net)

model.summary(input_shape=(1,32,32))
```

```python

```

**3，训练模型**

```python
from sklearn.metrics import accuracy_score

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu().numpy(),y_pred_cls.cpu().numpy()) 
    # 注意此处要将数据先移动到cpu上，然后才能转换成numpy数组

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # 注意此处compile时指定了device

dfhistory = model.fit(3,dl_train = dl_train, dl_val=dl_valid, log_step_freq=100) 

```

```
Start Training ...

================================================================================2020-06-27 00:24:29
{'step': 100, 'loss': 1.063, 'accuracy': 0.619}
{'step': 200, 'loss': 0.681, 'accuracy': 0.764}
{'step': 300, 'loss': 0.534, 'accuracy': 0.818}
{'step': 400, 'loss': 0.458, 'accuracy': 0.847}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   1   | 0.412 |  0.863   |  0.128   |    0.961     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-27 00:24:35
{'step': 100, 'loss': 0.147, 'accuracy': 0.956}
{'step': 200, 'loss': 0.156, 'accuracy': 0.954}
{'step': 300, 'loss': 0.156, 'accuracy': 0.954}
{'step': 400, 'loss': 0.157, 'accuracy': 0.955}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   2   | 0.153 |  0.956   |  0.085   |    0.976     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-27 00:24:42
{'step': 100, 'loss': 0.126, 'accuracy': 0.965}
{'step': 200, 'loss': 0.147, 'accuracy': 0.96}
{'step': 300, 'loss': 0.153, 'accuracy': 0.959}
{'step': 400, 'loss': 0.147, 'accuracy': 0.96}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   3   | 0.146 |   0.96   |  0.119   |    0.968     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-27 00:24:48
Finished Training...
```


**4，评估模型**

```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
```

```python
plot_metric(dfhistory, "loss")
```

```python
plot_metric(dfhistory,"accuracy")
```

```python
model.evaluate(dl_valid)
```

```
{'val_accuracy': 0.9603441455696202, 'val_loss': 0.14203246376371081}
```


**5，使用模型**

```python
model.predict(dl_valid)[0:10]
```

```
tensor([[ -9.2092,   3.1997,   1.4028,  -2.7135,  -0.7320,  -2.0518, -20.4938,
          14.6774,   1.7616,   5.8549],
        [  2.8509,   4.9781,  18.0946,   0.0928,  -1.6061,  -4.1437,   4.8697,
           3.8811,   4.3869,  -3.5929],
        [-22.5231,  13.6643,   5.0244, -11.0188, -16.8147,  -9.5894,  -6.2556,
         -10.5648, -12.1022, -19.4685],
        [ 23.2670, -12.0711,  -7.3968,  -8.2715,  -1.0915, -12.6050,   8.0444,
         -16.9339,   1.8827,  -0.2497],
        [ -4.1159,   3.2102,   0.4971, -11.8064,  12.1460,  -5.1650,  -6.5918,
           1.0088,   0.8362,   2.5132],
        [-26.1764,  15.6251,   6.1191, -12.2424, -13.9725, -10.0540,  -7.8669,
          -5.9602, -11.1944, -18.7890],
        [ -5.0602,   3.3779,  -0.6647,  -8.5185,  10.0320,  -5.5107,  -6.9579,
           2.3811,   0.2542,   3.2860],
        [  4.1017,  -0.4282,   7.2220,   3.3700,  -3.6813,   1.1576,  -1.8479,
           0.7450,   3.9768,   6.2640],
        [  1.9689,  -0.3960,   7.4414, -10.4789,   2.7066,   1.7482,   5.7971,
          -4.5808,   3.0911,  -5.1971],
        [ -2.9680,  -1.2369,  -0.0829,  -1.8577,   1.9380,  -0.8374,  -8.2207,
           3.5060,   3.8735,  13.6762]], device='cuda:0')
```


**6，保存模型**

```python
# save the model parameters
torch.save(model.net.module.state_dict(), "model_parameter.pkl")

net_clone = CnnModel()
net_clone.load_state_dict(torch.load("model_parameter.pkl"))

model_clone = torchkeras.Model(net_clone)
model_clone.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device)
model_clone.evaluate(dl_valid)

```

```
{'val_accuracy': 0.9603441455696202, 'val_loss': 0.14203246376371081}
```

```python

```
