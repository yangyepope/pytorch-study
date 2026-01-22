import torch
import torch.nn as nn

# 输入数据
x = torch.randn(6, 8)

target = torch.tensor([1, 2, 3, 4, 5, 6])
criterion = nn.CrossEntropyLoss()
output = criterion(x, target)
print(output)


# 使用独热编码作为目标
x = torch.randn(6, 8)
target = torch.randn(6, 8).softmax(dim=1)
loss = nn.CrossEntropyLoss()
loss(x, target)