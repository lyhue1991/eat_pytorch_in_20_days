
# 6-3,使用GPU训练模型


深度学习的训练过程常常非常耗时，一个模型训练几个小时是家常便饭，训练几天也是常有的事情，有时候甚至要训练几十天。

训练过程的耗时主要来自于两个部分，一部分来自数据准备，另一部分来自参数迭代。

当数据准备过程还是模型训练时间的主要瓶颈时，我们可以使用更多进程来准备数据。

当参数迭代过程成为训练时间的主要瓶颈时，我们通常的方法是应用GPU来进行加速。

```python
import torch 
import torchkeras 

print("torch.__version__ = ",torch.__version__)
print("torchkeras.__version__ = ",torchkeras.__version__)

```

<!-- #region -->
注：本节代码只能在有GPU的机器环境上才能正确执行。

对于没有GPU的同学，推荐使用kaggle平台上的GPU。


可点击如下链接，直接在kaggle中运行范例代码。

https://www.kaggle.com/lyhue1991/pytorch-gpu-examples


<!-- #endregion -->

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

## 〇，GPU相关操作汇总

```python
import torch 
from torch import nn 

# 1，查看gpu信息
if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)

gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)

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

```python
# 3，将模型中的全部张量移动到gpu上
net = nn.Linear(2,1)
print(next(net.parameters()).is_cuda)
net.to("cuda:0") # 将模型中的全部参数张量依次到GPU上，注意，无需重新赋值为 net = net.to("cuda:0")
print(next(net.parameters()).is_cuda)
print(next(net.parameters()).device)

```

```python
# 4，创建支持多个gpu数据并行的模型
linear = nn.Linear(2,1)
print(next(linear.parameters()).device)

model = nn.DataParallel(linear)
print(model.device_ids)
print(next(model.module.parameters()).device) 

#注意保存参数时要指定保存model.module的参数
torch.save(model.module.state_dict(), "model_parameter.pt") 

linear = nn.Linear(2,1)
linear.load_state_dict(torch.load("model_parameter.pt")) 

```

## 一，矩阵乘法范例


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

```python

```

## 二，线性回归范例


下面对比使用CPU和GPU训练一个线性回归模型的效率


### 1，使用CPU

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
loss_fn = nn.MSELoss()

def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        optimizer.zero_grad()
        Y_pred = linear(X) 
        loss = loss_fn(Y_pred,Y)
        loss.backward() 
        optimizer.step()
        if epoch%50==0:
            print({"epoch":epoch,"loss":loss.item()})
    toc = time.time()
    print("time used:",toc-tic)

train(500)
```

### 2，使用GPU

```python
# 准备数据
n = 1000000 #样本数量

X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0,-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0.t() + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动

# 数据移动到GPU上
print("torch.cuda.is_available() = ",torch.cuda.is_available())
X = X.cuda()
Y = Y.cuda()
print("X.device:",X.device)
print("Y.device:",Y.device)
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

```python
# 训练模型
optimizer = torch.optim.Adam(linear.parameters(),lr = 0.1)
loss_fn = nn.MSELoss()

def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        optimizer.zero_grad()
        Y_pred = linear(X) 
        loss = loss_fn(Y_pred,Y)
        loss.backward() 
        optimizer.step()
        if epoch%50==0:
            print({"epoch":epoch,"loss":loss.item()})
    toc = time.time()
    print("time used:",toc-tic)
    
train(500)
```

```python

```

## 三，图片分类范例

```python
import torch 
from torch import nn 

import torchvision 
from torchvision import transforms
```

```python
transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="minist/",train=True,download=True,transform=transform)
ds_val = torchvision.datasets.MNIST(root="minist/",train=False,download=True,transform=transform)

dl_train =  torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_val =  torch.utils.data.DataLoader(ds_val, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_val))

```

```python
def create_net():
    net = nn.Sequential()
    net.add_module("conv1",nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3))
    net.add_module("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2))
    net.add_module("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5))
    net.add_module("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2))
    net.add_module("dropout",nn.Dropout2d(p = 0.1))
    net.add_module("adaptive_pool",nn.AdaptiveMaxPool2d((1,1)))
    net.add_module("flatten",nn.Flatten())
    net.add_module("linear1",nn.Linear(64,32))
    net.add_module("relu",nn.ReLU())
    net.add_module("linear2",nn.Linear(32,10))
    return net

net = create_net()
print(net)
```

### 1，使用CPU进行训练

```python
import os,sys,time
import numpy as np
import pandas as pd
import datetime 
from tqdm import tqdm 

import torch
from torch import nn 
from copy import deepcopy
from torchmetrics import Accuracy
#注：多分类使用torchmetrics中的评估指标，二分类使用torchkeras.metrics中的评估指标

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(str(info)+"\n")
    

net = create_net() 

loss_fn = nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(net.parameters(),lr = 0.01)   
metrics_dict = {"acc":Accuracy()}

epochs = 20 
ckpt_path='checkpoint.pt'

#early_stopping相关设置
monitor="val_acc"
patience=5
mode="max"

history = {}

for epoch in range(1, epochs+1):
    printlog("Epoch {0} / {1}".format(epoch, epochs))

    # 1，train -------------------------------------------------  
    net.train()
    
    total_loss,step = 0,0
    
    loop = tqdm(enumerate(dl_train), total =len(dl_train))
    train_metrics_dict = deepcopy(metrics_dict) 
    
    for i, batch in loop: 
        
        features,labels = batch
        #forward
        preds = net(features)
        loss = loss_fn(preds,labels)
        
        #backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
        #metrics
        step_metrics = {"train_"+name:metric_fn(preds, labels).item() 
                        for name,metric_fn in train_metrics_dict.items()}
        
        step_log = dict({"train_loss":loss.item()},**step_metrics)

        total_loss += loss.item()
        
        step+=1
        if i!=len(dl_train)-1:
            loop.set_postfix(**step_log)
        else:
            epoch_loss = total_loss/step
            epoch_metrics = {"train_"+name:metric_fn.compute().item() 
                             for name,metric_fn in train_metrics_dict.items()}
            epoch_log = dict({"train_loss":epoch_loss},**epoch_metrics)
            loop.set_postfix(**epoch_log)

            for name,metric_fn in train_metrics_dict.items():
                metric_fn.reset()
                
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]
        

    # 2，validate -------------------------------------------------
    net.eval()
    
    total_loss,step = 0,0
    loop = tqdm(enumerate(dl_val), total =len(dl_val))
    
    val_metrics_dict = deepcopy(metrics_dict) 
    
    with torch.no_grad():
        for i, batch in loop: 

            features,labels = batch
            
            #forward
            preds = net(features)
            loss = loss_fn(preds,labels)

            #metrics
            step_metrics = {"val_"+name:metric_fn(preds, labels).item() 
                            for name,metric_fn in val_metrics_dict.items()}

            step_log = dict({"val_loss":loss.item()},**step_metrics)

            total_loss += loss.item()
            step+=1
            if i!=len(dl_val)-1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = (total_loss/step)
                epoch_metrics = {"val_"+name:metric_fn.compute().item() 
                                 for name,metric_fn in val_metrics_dict.items()}
                epoch_log = dict({"val_loss":epoch_loss},**epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name,metric_fn in val_metrics_dict.items():
                    metric_fn.reset()
                    
    epoch_log["epoch"] = epoch           
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]

    # 3，early-stopping -------------------------------------------------
    arr_scores = history[monitor]
    best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
    if best_score_idx==len(arr_scores)-1:
        torch.save(net.state_dict(),ckpt_path)
        print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
             arr_scores[best_score_idx]),file=sys.stderr)
    if len(arr_scores)-best_score_idx>patience:
        print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
            monitor,patience),file=sys.stderr)
        break 
    net.load_state_dict(torch.load(ckpt_path))
    
dfhistory = pd.DataFrame(history)

```

<!-- #region -->
================================================================================2022-07-17 15:07:03
Epoch 1 / 20

100%|██████████| 469/469 [00:57<00:00,  8.15it/s, train_acc=0.909, train_loss=0.279] 
100%|██████████| 79/79 [00:04<00:00, 16.80it/s, val_acc=0.956, val_loss=0.147] 

================================================================================2022-07-17 15:08:06
Epoch 2 / 20


<<<<<< reach best val_acc : 0.9556000232696533 >>>>>>
100%|██████████| 469/469 [00:58<00:00,  8.03it/s, train_acc=0.968, train_loss=0.105] 
100%|██████████| 79/79 [00:04<00:00, 18.59it/s, val_acc=0.977, val_loss=0.0849]

================================================================================2022-07-17 15:09:09
Epoch 3 / 20


<<<<<< reach best val_acc : 0.9765999913215637 >>>>>>
100%|██████████| 469/469 [00:58<00:00,  8.07it/s, train_acc=0.974, train_loss=0.0882]
100%|██████████| 79/79 [00:04<00:00, 17.13it/s, val_acc=0.984, val_loss=0.0554] 
<<<<<< reach best val_acc : 0.9843000173568726 >>>>>>

================================================================================2022-07-17 15:10:12
Epoch 4 / 20

100%|██████████| 469/469 [01:01<00:00,  7.63it/s, train_acc=0.976, train_loss=0.0814] 
100%|██████████| 79/79 [00:04<00:00, 16.34it/s, val_acc=0.979, val_loss=0.0708]

================================================================================2022-07-17 15:11:18
Epoch 5 / 20


100%|██████████| 469/469 [01:03<00:00,  7.42it/s, train_acc=0.974, train_loss=0.0896]
100%|██████████| 79/79 [00:05<00:00, 14.06it/s, val_acc=0.979, val_loss=0.076] 

================================================================================2022-07-17 15:12:28
Epoch 6 / 20


100%|██████████| 469/469 [01:00<00:00,  7.77it/s, train_acc=0.972, train_loss=0.0937]
100%|██████████| 79/79 [00:04<00:00, 17.45it/s, val_acc=0.976, val_loss=0.0787] 

================================================================================2022-07-17 15:13:33
Epoch 7 / 20


100%|██████████| 469/469 [01:01<00:00,  7.63it/s, train_acc=0.974, train_loss=0.0858]
100%|██████████| 79/79 [00:05<00:00, 14.50it/s, val_acc=0.976, val_loss=0.082] 

================================================================================2022-07-17 15:14:40
Epoch 8 / 20


100%|██████████| 469/469 [00:59<00:00,  7.85it/s, train_acc=0.972, train_loss=0.0944]
100%|██████████| 79/79 [00:04<00:00, 17.21it/s, val_acc=0.982, val_loss=0.062] 
<<<<<< val_acc without improvement in 5 epoch, early stopping >>>>>>

<!-- #endregion -->

CPU每个Epoch大概1分钟


### 2，使用GPU进行训练

```python
import os,sys,time
import numpy as np
import pandas as pd
import datetime 
from tqdm import tqdm 

import torch
from torch import nn 
from copy import deepcopy
from torchmetrics import Accuracy
#注：多分类使用torchmetrics中的评估指标，二分类使用torchkeras.metrics中的评估指标

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(str(info)+"\n")
    
net = create_net() 


loss_fn = nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(net.parameters(),lr = 0.01)   
metrics_dict = {"acc":Accuracy()}


# =========================移动模型到GPU上==============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
loss_fn.to(device)
for name,fn in metrics_dict.items():
    fn.to(device)
# ====================================================================


epochs = 20 
ckpt_path='checkpoint.pt'

#early_stopping相关设置
monitor="val_acc"
patience=5
mode="max"

history = {}

for epoch in range(1, epochs+1):
    printlog("Epoch {0} / {1}".format(epoch, epochs))

    # 1，train -------------------------------------------------  
    net.train()
    
    total_loss,step = 0,0
    
    loop = tqdm(enumerate(dl_train), total =len(dl_train))
    train_metrics_dict = deepcopy(metrics_dict) 
    
    for i, batch in loop: 
        
        features,labels = batch
        
        # =========================移动数据到GPU上==============================
        features = features.to(device)
        labels = labels.to(device)
        # ====================================================================
        
        #forward
        preds = net(features)
        loss = loss_fn(preds,labels)
        
        #backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
        #metrics
        step_metrics = {"train_"+name:metric_fn(preds, labels).item() 
                        for name,metric_fn in train_metrics_dict.items()}
        
        step_log = dict({"train_loss":loss.item()},**step_metrics)

        total_loss += loss.item()
        
        step+=1
        if i!=len(dl_train)-1:
            loop.set_postfix(**step_log)
        else:
            epoch_loss = total_loss/step
            epoch_metrics = {"train_"+name:metric_fn.compute().item() 
                             for name,metric_fn in train_metrics_dict.items()}
            epoch_log = dict({"train_loss":epoch_loss},**epoch_metrics)
            loop.set_postfix(**epoch_log)

            for name,metric_fn in train_metrics_dict.items():
                metric_fn.reset()
                
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]
        

    # 2，validate -------------------------------------------------
    net.eval()
    
    total_loss,step = 0,0
    loop = tqdm(enumerate(dl_val), total =len(dl_val))
    
    val_metrics_dict = deepcopy(metrics_dict) 
    
    with torch.no_grad():
        for i, batch in loop: 

            features,labels = batch
            
            # =========================移动数据到GPU上==============================
            features = features.to(device)
            labels = labels.to(device)
            # ====================================================================
            
            #forward
            preds = net(features)
            loss = loss_fn(preds,labels)

            #metrics
            step_metrics = {"val_"+name:metric_fn(preds, labels).item() 
                            for name,metric_fn in val_metrics_dict.items()}

            step_log = dict({"val_loss":loss.item()},**step_metrics)

            total_loss += loss.item()
            step+=1
            if i!=len(dl_val)-1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = (total_loss/step)
                epoch_metrics = {"val_"+name:metric_fn.compute().item() 
                                 for name,metric_fn in val_metrics_dict.items()}
                epoch_log = dict({"val_loss":epoch_loss},**epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name,metric_fn in val_metrics_dict.items():
                    metric_fn.reset()
                    
    epoch_log["epoch"] = epoch           
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]

    # 3，early-stopping -------------------------------------------------
    arr_scores = history[monitor]
    best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
    if best_score_idx==len(arr_scores)-1:
        torch.save(net.state_dict(),ckpt_path)
        print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
             arr_scores[best_score_idx]),file=sys.stderr)
    if len(arr_scores)-best_score_idx>patience:
        print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
            monitor,patience),file=sys.stderr)
        break 
    net.load_state_dict(torch.load(ckpt_path))
    
dfhistory = pd.DataFrame(history)

```

```
================================================================================2022-07-17 15:20:40
Epoch 1 / 20

100%|██████████| 469/469 [00:12<00:00, 37.07it/s, train_acc=0.89, train_loss=0.336]  
100%|██████████| 79/79 [00:02<00:00, 37.31it/s, val_acc=0.95, val_loss=0.16]   

================================================================================2022-07-17 15:20:55
Epoch 2 / 20


<<<<<< reach best val_acc : 0.9498000144958496 >>>>>>
100%|██████████| 469/469 [00:12<00:00, 37.04it/s, train_acc=0.964, train_loss=0.115] 
100%|██████████| 79/79 [00:01<00:00, 43.36it/s, val_acc=0.972, val_loss=0.0909]

================================================================================2022-07-17 15:21:10
Epoch 3 / 20


<<<<<< reach best val_acc : 0.9721999764442444 >>>>>>
100%|██████████| 469/469 [00:12<00:00, 38.05it/s, train_acc=0.971, train_loss=0.0968]
100%|██████████| 79/79 [00:01<00:00, 42.10it/s, val_acc=0.974, val_loss=0.0878] 

================================================================================2022-07-17 15:21:24
Epoch 4 / 20

<<<<<< reach best val_acc : 0.974399983882904 >>>>>>
100%|██████████| 469/469 [00:13<00:00, 35.56it/s, train_acc=0.973, train_loss=0.089] 
100%|██████████| 79/79 [00:02<00:00, 38.16it/s, val_acc=0.982, val_loss=0.0585]

================================================================================2022-07-17 15:21:40
Epoch 5 / 20


<<<<<< reach best val_acc : 0.9822999835014343 >>>>>>
100%|██████████| 469/469 [00:12<00:00, 36.80it/s, train_acc=0.977, train_loss=0.0803]
100%|██████████| 79/79 [00:01<00:00, 42.38it/s, val_acc=0.976, val_loss=0.0791]

================================================================================2022-07-17 15:21:55
Epoch 6 / 20


100%|██████████| 469/469 [00:13<00:00, 34.63it/s, train_acc=0.977, train_loss=0.0787]
100%|██████████| 79/79 [00:02<00:00, 39.01it/s, val_acc=0.97, val_loss=0.105]   

================================================================================2022-07-17 15:22:11
Epoch 7 / 20


100%|██████████| 469/469 [00:12<00:00, 37.39it/s, train_acc=0.975, train_loss=0.0871]
100%|██████████| 79/79 [00:02<00:00, 39.16it/s, val_acc=0.984, val_loss=0.0611]

================================================================================2022-07-17 15:22:26
Epoch 8 / 20


<<<<<< reach best val_acc : 0.9835000038146973 >>>>>>
100%|██████████| 469/469 [00:13<00:00, 35.63it/s, train_acc=0.976, train_loss=0.0774] 
100%|██████████| 79/79 [00:01<00:00, 42.92it/s, val_acc=0.982, val_loss=0.0778] 

================================================================================2022-07-17 15:22:41
Epoch 9 / 20


100%|██████████| 469/469 [00:12<00:00, 37.96it/s, train_acc=0.976, train_loss=0.0819]
100%|██████████| 79/79 [00:01<00:00, 42.99it/s, val_acc=0.981, val_loss=0.0652] 

================================================================================2022-07-17 15:22:56
Epoch 10 / 20


100%|██████████| 469/469 [00:13<00:00, 35.29it/s, train_acc=0.975, train_loss=0.0852]
100%|██████████| 79/79 [00:01<00:00, 41.38it/s, val_acc=0.978, val_loss=0.0808]

================================================================================2022-07-17 15:23:12
Epoch 11 / 20


100%|██████████| 469/469 [00:12<00:00, 38.77it/s, train_acc=0.975, train_loss=0.0863] 
100%|██████████| 79/79 [00:01<00:00, 42.71it/s, val_acc=0.983, val_loss=0.0665] 

================================================================================2022-07-17 15:23:26
Epoch 12 / 20


100%|██████████| 469/469 [00:12<00:00, 36.55it/s, train_acc=0.976, train_loss=0.0818]
100%|██████████| 79/79 [00:02<00:00, 37.44it/s, val_acc=0.979, val_loss=0.0819]
<<<<<< val_acc without improvement in 5 epoch, early stopping >>>>>>
```


使用GPU后每个Epoch只需要10秒钟左右，提升了6倍。



## 四，torchkeras.KerasModel中使用GPU

```python
从上面的例子可以看到，在pytorch中使用GPU并不复杂，但对于经常炼丹的同学来说，模型和数据老是移来移去还是蛮麻烦的。

一不小心就会忘了移动某些数据或者某些module，导致报错。

torchkeras.KerasModel 在设计的时候考虑到了这一点，如果环境当中存在可用的GPU，会自动使用GPU，反之则使用CPU。

通过引入accelerate的一些基础功能，torchkeras.KerasModel以非常优雅的方式在GPU和CPU之间切换。

详细实现可以参考torchkeras.KerasModel的源码。
```

```python
!pip install torchkeras==3.2.3
```

```python
import  accelerate 
accelerator = accelerate.Accelerator()
print(accelerator.device)  
```

```python
from torchkeras import KerasModel 
from torchmetrics import Accuracy

net = create_net() 
model = KerasModel(net,
                   loss_fn=nn.CrossEntropyLoss(),
                   metrics_dict = {"acc":Accuracy()},
                   optimizer = torch.optim.Adam(net.parameters(),lr = 0.01)  )

model.fit(
    train_data = dl_train,
    val_data= dl_val,
    epochs=10,
    patience=3,
    monitor="val_acc", 
    mode="max")
```

```python

```

## 五，torchkeras.LightModel中使用GPU


通过引用pytorch_lightning的功能，

torchkeras.LightModel以更加显式的方式支持GPU训练，

不仅如此，还能支持多GPU和TPU训练。


```python
from torchmetrics import Accuracy 
from torchkeras import LightModel 

net = create_net() 
model = LightModel(net,
                   loss_fn=nn.CrossEntropyLoss(),
                   metrics_dict = {"acc":Accuracy()},
                   optimizer = torch.optim.Adam(net.parameters(),lr = 0.01) )

```

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
                     gpus=1,
                     callbacks = [model_ckpt,early_stopping],
                     enable_progress_bar = True) 


##4，启动训练循环
trainer.fit(model,dl_train,dl_val)


```

```
================================================================================2022-07-18 00:18:14
{'epoch': 0, 'val_loss': 2.31911301612854, 'val_acc': 0.0546875}
<<<<<< reach best val_acc : 0.0546875 >>>>>>

================================================================================2022-07-18 00:18:29
{'epoch': 0, 'val_loss': 0.10364170372486115, 'val_acc': 0.9693999886512756}
{'epoch': 0, 'train_loss': 0.31413567066192627, 'train_acc': 0.8975499868392944}
<<<<<< reach best val_acc : 0.9693999886512756 >>>>>>

================================================================================2022-07-18 00:18:43
{'epoch': 1, 'val_loss': 0.0983758345246315, 'val_acc': 0.9710999727249146}
{'epoch': 1, 'train_loss': 0.10680060088634491, 'train_acc': 0.9673333168029785}
<<<<<< reach best val_acc : 0.9710999727249146 >>>>>>

================================================================================2022-07-18 00:18:58
{'epoch': 2, 'val_loss': 0.08315123617649078, 'val_acc': 0.9764999747276306}
{'epoch': 2, 'train_loss': 0.09339822083711624, 'train_acc': 0.9722166657447815}
<<<<<< reach best val_acc : 0.9764999747276306 >>>>>>

================================================================================2022-07-18 00:19:13
{'epoch': 3, 'val_loss': 0.06529796123504639, 'val_acc': 0.9799000024795532}
{'epoch': 3, 'train_loss': 0.08487282693386078, 'train_acc': 0.9746000170707703}
<<<<<< reach best val_acc : 0.9799000024795532 >>>>>>

================================================================================2022-07-18 00:19:27
{'epoch': 4, 'val_loss': 0.10162600129842758, 'val_acc': 0.9735000133514404}
{'epoch': 4, 'train_loss': 0.08439336717128754, 'train_acc': 0.9746666550636292}

================================================================================2022-07-18 00:19:42
{'epoch': 5, 'val_loss': 0.0818500965833664, 'val_acc': 0.9789000153541565}
{'epoch': 5, 'train_loss': 0.08107426762580872, 'train_acc': 0.9763166904449463}

================================================================================2022-07-18 00:19:56
{'epoch': 6, 'val_loss': 0.08046088367700577, 'val_acc': 0.979200005531311}
{'epoch': 6, 'train_loss': 0.08173364400863647, 'train_acc': 0.9772833585739136}
```


**如果本书对你有所帮助，想鼓励一下作者，记得给本项目加一颗星星star⭐️，并分享给你的朋友们喔😊!** 

如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"算法美食屋"下留言。作者时间和精力有限，会酌情予以回复。

也可以在公众号后台回复关键字：**加群**，加入读者交流群和大家讨论。

![算法美食屋logo.png](https://tva1.sinaimg.cn/large/e6c9d24egy1h41m2zugguj20k00b9q46.jpg)
