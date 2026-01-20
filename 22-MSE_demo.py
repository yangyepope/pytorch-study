import torch
import torch.nn as nn

# 定义 MSE 损失函数
criterion = nn.MSELoss()

# 假设预测值和真实值
y_pred = torch.tensor([0.5, 1.2, 2.0])
y_true = torch.tensor([0.0, 1.0, 2.5])

# 计算 MSE
loss = criterion(y_pred, y_true)
print(loss) # 输出: ((0.5^2 + 0.2^2 + 0.5^2) / 3) = 0.18