import torch
import torch.nn as nn

# 1. 创建线性层并手动指定参数，方便计算
layer = nn.Linear(1, 1)
layer.weight.data = torch.tensor([[2.0]]) # w=2
layer.bias.data = torch.tensor([[0.5]])   # b=0.5

# 2. 前向传播
x = torch.tensor([[1.0]])
output = layer(x) # y = 2*1 + 0.5 = 2.5

# 3. 假设目标值是 5.0，计算 MSE Loss
target = torch.tensor([[5.0]])
loss = (output - target)**2 # (2.5 - 5.0)^2 = 6.25

# 4. 反向传播
loss.backward()

# 5. 查看偏置的梯度
print(f"偏置 b 的当前值: {layer.bias.data.item()}")
print(f"偏置 b 的梯度: {layer.bias.grad.item()}")
# 根据公式 2*(2.5-5.0)*1 = -5.0