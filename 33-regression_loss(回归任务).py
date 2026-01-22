import torch
import torch.nn as nn


# 定义数据
x = torch.randn(6, 8)

# 定义目标值
target = torch.randn(6, 8)


# 计算均方误差损失 MSE
mse_loss = nn.MSELoss()
output = mse_loss(x, target)
print(output)


# 计算平均绝对误差损失 MAE
mae_loss = nn.L1Loss()
output = mae_loss(x, target)
print(output)

# 计算平滑L1损失 Smooth L1 Loss
smooth_l1_loss = nn.SmoothL1Loss()
output = smooth_l1_loss(x, target)
print(output)
