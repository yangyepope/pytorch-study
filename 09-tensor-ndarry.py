

# tensor ->  ndarray
import torch
import numpy as np
np.set_printoptions(precision=6)
torch.set_printoptions(precision=6)

rand1 = torch.rand(2, 3)
print(rand1)

# tensor -> ndarry
ndarry1 = rand1.numpy()
print(ndarry1)


rand1[:,0] = 10
print(rand1)
print(ndarry1)

ndarry1[:,0] = 20
print(rand1)
print(ndarry1)


# 避免共享内存，使用copy方法避免共享内存

ndarry2 = rand1.numpy().copy()
print(ndarry2)


rand1[:,0] = 5
print(rand1)
print(ndarry2)
print(ndarry1)


# ndarry -> tensor

npr1 = np.random.rand(2, 3)
print(npr1)
rand2 = torch.from_numpy(npr1)
print(rand2)