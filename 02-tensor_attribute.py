import torch


dev = torch.device('cpu')

a = torch.tensor([2, 3],dtype=torch.float32 ,device=dev)
print(a)


# 稀疏张量
i = torch.tensor([[0,1,2],[0,1,2]])
v = torch.tensor([1,2,3])
a = torch.sparse_coo_tensor(i,v,size=(4,4))
print(a)

# 稀疏张量转稠密张量
dense = torch.sparse_coo_tensor(i, v, size=(4, 4), dtype=torch.float32).to_dense()
print(dense)
