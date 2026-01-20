import torch
import torch.nn as nn

# 定义一个输入 3，输出 2 的层
layer = nn.Linear(3, 2)

# 查看偏置的形状
print(f"偏置的形状: {layer.bias.shape}")
# 输出: torch.Size([2])  <-- 对应 out_features

# 即使我们输入 100 行数据
input_data = torch.randn(100, 3)
output = layer(input_data)

print(f"输出的形状: {output.shape}")
# 输出: torch.Size([100, 2]) <-- 行数变了，但每一行用的都是那 2 个偏置