<!-- #region -->
# 3-3,高阶API示范

Pytorch没有官方的高阶API，一般需要用户自己实现训练循环、验证循环、和预测循环。

作者通过仿照keras的功能对Pytorch的nn.Module进行了封装，设计了torchkeras.KerasModel类，

实现了 fit, evaluate，predict等方法，相当于用户自定义高阶API。

并示范了用它实现线性回归模型。

此外，作者还通过借用pytorch_lightning的功能，封装了类Keras接口的另外一种实现，即torchkeras.LightModel类。

并示范了用它实现DNN二分类模型。


torchkeras.KerasModel类和torchkeras.LightModel类看起来非常强大，但实际上它们的源码非常简单，不足200行。
我们在第一章中`一、Pytorch的建模流程`用到的训练代码其实就是torchkeras库的核心源码。

在实际应用中，由于有些模型的输入输出以及Loss结构和torchkeras的假设结构有所不同，直接调用torchkeras可能不能满足需求，不要害怕，copy出来
torchkeras.KerasModel或者torchkeras.LightModel的源码，在输入输出和Loss上简单改动一下就可以。

<!-- #endregion -->

```python

```

```python
import os
import datetime

#打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

#mac系统上pytorch和matplotlib在jupyter中同时跑需要更改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

```

```python
!pip install torchkeras==3.2.2 -i https://pypi.python.org/simple
```

```python
import torch 
import torchkeras 


print("torch.__version__="+torch.__version__) 
print("torchkeras.__version__="+torchkeras.__version__) 
```

```
torch.__version__=1.10.0
torchkeras.__version__=3.2.2
```

```python

```

### 一，线性回归模型


此范例我们通过继承torchkeras.Model模型接口，实现线性回归模型。


**1，准备数据**

```python
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset

#样本数量
n = 400

# 生成测试用数据集
X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0],[-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动

```

```python
# 数据可视化

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0],Y[:,0], c = "b",label = "samples")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)

ax2 = plt.subplot(122)
ax2.scatter(X[:,1],Y[:,0], c = "g",label = "samples")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)
plt.show()

```

![](./data/3-3-回归数据可视化.png)

```python
#构建输入数据管道
ds = TensorDataset(X,Y)
ds_train,ds_val = torch.utils.data.random_split(ds,[int(400*0.7),400-int(400*0.7)])
dl_train = DataLoader(ds_train,batch_size = 10,shuffle=True,num_workers=2)
dl_val = DataLoader(ds_val,batch_size = 10,num_workers=2)

features,labels = next(iter(dl_train))

```

```python

```

**2，定义模型**

```python
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(2,1)
    
    def forward(self,x):
        return self.fc(x)

net = LinearRegression()

```

```python

from torchkeras import summary 

summary(net,input_data=features);
```

```
--------------------------------------------------------------------------
Layer (type)                            Output Shape              Param #
==========================================================================
Linear-1                                     [-1, 1]                    3
==========================================================================
Total params: 3
Trainable params: 3
Non-trainable params: 0
--------------------------------------------------------------------------
Input size (MB): 0.000069
Forward/backward pass size (MB): 0.000008
Params size (MB): 0.000011
Estimated Total Size (MB): 0.000088
--------------------------------------------------------------------------
```

```python

```

**3，训练模型**

```python
from torchkeras import KerasModel 

import torchmetrics

net = LinearRegression()
model = KerasModel(net=net,
                   loss_fn = nn.MSELoss(),
                   metrics_dict = {"mae":torchmetrics.MeanAbsoluteError()},
                   optimizer= torch.optim.Adam(net.parameters(),lr = 0.05))

dfhistory = model.fit(train_data=dl_train,
      val_data=dl_val,
      epochs=20,
      ckpt_path='checkpoint.pt',
      patience=5,
      monitor='val_loss',
      mode='min')

```

```python
# 结果可视化

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

w,b = net.state_dict()["fc.weight"],net.state_dict()["fc.bias"]

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0],Y[:,0], c = "b",label = "samples")
ax1.plot(X[:,0],w[0,0]*X[:,0]+b[0],"-r",linewidth = 5.0,label = "model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)


ax2 = plt.subplot(122)
ax2.scatter(X[:,1],Y[:,0], c = "g",label = "samples")
ax2.plot(X[:,1],w[0,1]*X[:,1]+b[0],"-r",linewidth = 5.0,label = "model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)

plt.show()

```

**4，评估模型**

```python
dfhistory.tail()
```

```
	train_loss	train_mae	val_loss	val_mae	epoch
15	4.339620	1.635648	3.119237	1.384351	16
16	4.313104	1.631849	2.999482	1.352427	17
17	4.319547	1.628811	3.022779	1.355054	18
18	4.315403	1.636815	3.087339	1.369488	19
19	4.266822	1.627701	2.937751	1.330670	20
```

```python

```

```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory["train_"+metric]
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

![](./data/3-3-loss曲线.png)

```python
plot_metric(dfhistory,"mae")
```

![](./data/3-3-mape曲线.png)

```python
# 评估
model.evaluate(dl_val)
```

```
{'val_loss': 2.9377514322598777, 'val_mae': 1.3306695222854614}
```

```python

```

**5，使用模型**

```python
# 预测
dl = DataLoader(TensorDataset(X))
model.predict(dl)[0:10]
```

```
tensor([[ 3.9212],
        [ 8.6336],
        [ 6.1982],
        [ 6.1212],
        [-5.0974],
        [-6.3183],
        [ 4.6588],
        [ 5.5349],
        [11.9106],
        [24.6937]])
```

```python
# 预测
model.predict(dl_val)[0:10]
```

```
tensor([[-11.0324],
        [ 26.2708],
        [ 24.8866],
        [ 12.2698],
        [-12.0984],
        [  6.7254],
        [ 12.8081],
        [ 20.6977],
        [  5.4715],
        [  2.0188]])
```

```python

```

### 二，DNN二分类模型


此范例我们通过继承torchkeras.LightModel模型接口，实现DNN二分类模型。



**1，准备数据**

```python
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torchkeras 
import pytorch_lightning as pl 
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

#正负样本数量
n_positive,n_negative = 2000,2000

#生成正样本, 小圆环分布
r_p = 5.0 + torch.normal(0.0,1.0,size = [n_positive,1]) 
theta_p = 2*np.pi*torch.rand([n_positive,1])
Xp = torch.cat([r_p*torch.cos(theta_p),r_p*torch.sin(theta_p)],axis = 1)
Yp = torch.ones_like(r_p)

#生成负样本, 大圆环分布
r_n = 8.0 + torch.normal(0.0,1.0,size = [n_negative,1]) 
theta_n = 2*np.pi*torch.rand([n_negative,1])
Xn = torch.cat([r_n*torch.cos(theta_n),r_n*torch.sin(theta_n)],axis = 1)
Yn = torch.zeros_like(r_n)

#汇总样本
X = torch.cat([Xp,Xn],axis = 0)
Y = torch.cat([Yp,Yn],axis = 0)


#可视化
plt.figure(figsize = (6,6))
plt.scatter(Xp[:,0],Xp[:,1],c = "r")
plt.scatter(Xn[:,0],Xn[:,1],c = "g")
plt.legend(["positive","negative"]);

```

![](./data/3-3-分类数据可视化.png)

```python
ds = TensorDataset(X,Y)

ds_train,ds_val = torch.utils.data.random_split(ds,[int(len(ds)*0.7),len(ds)-int(len(ds)*0.7)])
dl_train = DataLoader(ds_train,batch_size = 100,shuffle=True,num_workers=2)
dl_val = DataLoader(ds_val,batch_size = 100,num_workers=2)

```

```python

```

**2，定义模型**

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,8) 
        self.fc3 = nn.Linear(8,1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y
    
```

```python
import torchkeras 
from torchkeras.metrics import Accuracy 

net = Net()
loss_fn = nn.BCEWithLogitsLoss()
metric_dict = {"acc":Accuracy()}

optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.0001)

model = torchkeras.LightModel(net,
                   loss_fn = loss_fn,
                   metrics_dict= metric_dict,
                   optimizer = optimizer,
                   lr_scheduler = lr_scheduler,
                  )       

from torchkeras import summary
summary(model,input_data=features);

```

```python

```

**3，训练模型**

```python
import pytorch_lightning as pl     

#1，设置回调函数
model_ckpt = pl.callbacks.ModelCheckpoint(
    monitor='val_acc',
    save_top_k=1,
    mode='max'
)

early_stopping = pl.callbacks.EarlyStopping(monitor = 'val_acc',
                           patience=3,
                           mode = 'max'
                          )

#2，设置训练参数

# gpus=0 则使用cpu训练，gpus=1则使用1个gpu训练，gpus=2则使用2个gpu训练，gpus=-1则使用所有gpu训练，
# gpus=[0,1]则指定使用0号和1号gpu训练， gpus="0,1,2,3"则使用0,1,2,3号gpu训练
# tpus=1 则使用1个tpu训练
trainer = pl.Trainer(logger=True,
                     min_epochs=3,max_epochs=20,
                     gpus=0,
                     callbacks = [model_ckpt,early_stopping],
                     enable_progress_bar = True) 


##4，启动训练循环
trainer.fit(model,dl_train,dl_val)


```

```
================================================================================2022-07-16 20:25:49
{'epoch': 0, 'val_loss': 0.3484574854373932, 'val_acc': 0.8766666650772095}
{'epoch': 0, 'train_loss': 0.5639581680297852, 'train_acc': 0.708214282989502}
<<<<<< reach best val_acc : 0.8766666650772095 >>>>>>

================================================================================2022-07-16 20:25:54
{'epoch': 1, 'val_loss': 0.18654096126556396, 'val_acc': 0.925000011920929}
{'epoch': 1, 'train_loss': 0.2512527406215668, 'train_acc': 0.9117857217788696}
<<<<<< reach best val_acc : 0.925000011920929 >>>>>>

================================================================================2022-07-16 20:25:59
{'epoch': 2, 'val_loss': 0.19609291851520538, 'val_acc': 0.9191666841506958}
{'epoch': 2, 'train_loss': 0.19115397334098816, 'train_acc': 0.9257143139839172}

================================================================================2022-07-16 20:26:04
{'epoch': 3, 'val_loss': 0.18749761581420898, 'val_acc': 0.925000011920929}
{'epoch': 3, 'train_loss': 0.19545568525791168, 'train_acc': 0.9235714077949524}

================================================================================2022-07-16 20:26:09
{'epoch': 4, 'val_loss': 0.21518440544605255, 'val_acc': 0.9083333611488342}
{'epoch': 4, 'train_loss': 0.1998639553785324, 'train_acc': 0.9192857146263123}
```


```python
# 结果可视化
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (12,5))
ax1.scatter(Xp[:,0],Xp[:,1], c="r")
ax1.scatter(Xn[:,0],Xn[:,1],c = "g")
ax1.legend(["positive","negative"]);
ax1.set_title("y_true");

Xp_pred = X[torch.squeeze(net.forward(X)>=0.5)]
Xn_pred = X[torch.squeeze(net.forward(X)<0.5)]

ax2.scatter(Xp_pred[:,0],Xp_pred[:,1],c = "r")
ax2.scatter(Xn_pred[:,0],Xn_pred[:,1],c = "g")
ax2.legend(["positive","negative"]);
ax2.set_title("y_pred");

```

![](./data/3-3-分类结果可视化.png)


**4，评估模型**

```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory["train_"+metric]
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
dfhistory  = model.get_history() 
plot_metric(dfhistory,"loss")

```

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h491k7wtl0j20f70a6wes.jpg)

```python

```

```python
plot_metric(dfhistory,"acc")
```

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h491k8if3hj20ev0aaglw.jpg)

```python
#使用最佳保存点进行评估
trainer.test(ckpt_path='best', dataloaders=dl_val,verbose = False)
```

```
{'test_loss': 0.18654096126556396, 'test_acc': 0.925000011920929}
```

```python

```

**5，使用模型**

```python
predictions = F.sigmoid(torch.cat(trainer.predict(model, dl_val, ckpt_path='best'))) 
predictions 
```

```
tensor([[0.3873],
        [0.0028],
        [0.8772],
        ...,
        [0.9886],
        [0.4970],
        [0.8094]])
```

```python
def predict(model,dl):
    model.eval()
    result = torch.cat([model.forward(t[0]) for t in dl])
    return(result.data)

print(model.device)
predictions = F.sigmoid(predict(model,dl_val)[:10]) 
```

```python

```

**6，保存模型**

```python
print(trainer.checkpoint_callback.best_model_path)
print(trainer.checkpoint_callback.best_model_score)
```

```python
model_loaded = torchkeras.LightModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
```

```python

```

```python

```

**如果本书对你有所帮助，想鼓励一下作者，记得给本项目加一颗星星star⭐️，并分享给你的朋友们喔😊!** 

如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"算法美食屋"下留言。作者时间和精力有限，会酌情予以回复。

也可以在公众号后台回复关键字：**加群**，加入读者交流群和大家讨论。

![算法美食屋logo.png](https://tva1.sinaimg.cn/large/e6c9d24egy1h41m2zugguj20k00b9q46.jpg)
