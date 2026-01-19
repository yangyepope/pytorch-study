import torch

x = torch.rand(3, 5)
print(x)

y = torch.softmax(x, dim=1)
print(y)
