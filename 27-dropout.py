import torch

import torch.nn as nn

# 定义一个包含 Dropout 层的简单神经网络

x = torch.randint(1, 10, (10,), dtype=torch.float32)

dropout = torch.nn.Dropout(p=0.5)

y = dropout(x)
print("dropout前:",x)
print("dropout后:",y)
