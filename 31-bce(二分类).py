import torch
import torch.nn as nn


# 输入数据
input_data = torch.randn(3,2,requires_grad= True)
print(input_data)


#手动设置权重和偏置
weights = torch.tensor([[2.0, 1.0]],dtype=torch.float32,requires_grad=True)
bias = torch.tensor([[-1.0]],dtype=torch.float32,requires_grad=True)



# 使用手动设置权重和偏置
linear = torch.nn.functional.linear(input_data, weights, bias)
# 使用激活函数
pred = torch.sigmoid(linear)


# 目标值
target = torch.tensor([[1],[0],[1]],dtype=torch.float32)

# 定义二分类交叉熵损失函数
bce_loss = nn.BCELoss()

# 计算损失
loss = bce_loss(pred, target)

print("loss:",loss)

# 反向传播
loss.backward()
print(input_data.grad)
print(weights.grad)
print(bias.grad)
