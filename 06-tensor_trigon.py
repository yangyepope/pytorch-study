import torch

rand = torch.rand(2, 3)
zeros = torch.zeros(2, 3)
cos = torch.cos(rand)
print(rand)
print(cos)
torch_cos = torch.cos(zeros)
print(torch_cos)
