import numpy as np
import torch


# 张量的转换
rand = torch.rand(3)
print(rand)

tensor1 = torch.tensor([1, 2, 3])
print( tensor1)

tensor__type = tensor1.type(torch.float32)
print(tensor__type)
print(tensor__type.dtype)

half = tensor1.half()
print(half.dtype)

to = half.to(torch.complex32)
print(to.dtype)
