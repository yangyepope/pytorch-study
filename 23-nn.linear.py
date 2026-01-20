import torch
import torch.nn as nn

# 1. 定义线性层：输入维度 3，输出维度 2
model = nn.Linear(3, 2)

# 2. 模拟输入数据：假设有 5 个样本（Batch Size = 5）
# 形状为 [5, 3] -> [样本数, 特征数]

# 输入的维度必须和特征输入的维度一致
input_data = torch.randn(5, 3)

# 3. 前向传播
output = model(input_data)

# 4. 查看结果
print(f"输入形状: {input_data.shape}") # torch.Size([5, 3])
print(f"输出形状: {output.shape}")    # torch.Size([5, 2])