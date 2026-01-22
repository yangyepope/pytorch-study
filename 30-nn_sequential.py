import torch
import torch.nn as nn
from torchsummary import summary


# 1.定义数据
x = torch.randn(10, 3)  # 假设有10个样本，每个样本有3个特征

# 2.定义神经网络模型，使用nn.Sequential
model = nn.Sequential(
    nn.Linear(in_features=3, out_features=4 ),  # 第一个线性
    nn.Tanh(),
    nn.Linear(in_features=4, out_features=4),  # 第2个线性
    nn.ReLU(),
    nn.Linear(in_features=4, out_features=2),  # 第3个线性
    nn.Softmax(dim=1)
     )


# 3. 定义一个参数初始化函数
def init_paras(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


# 4. 参数初始化
model.apply(init_paras)

# 5.前向传播
output = model(x)
print(output)

# 6.查看模型参数
print("==========first layer==============")
summary(model,input_size=(3,),batch_size=10,device='cpu')

