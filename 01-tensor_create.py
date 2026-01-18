import torch

a = torch.Tensor([[1, 2], [3, 4]])
print(a)
print(a.shape)
print(a.dtype)
print(a.device)
print(a.type())

print("=========================")

a = torch.Tensor(2,3)
print(a)
print(a.shape)
print(a.type())

print("=========================")
a = torch.eye(2,2)
print(a)
print(a.type)

print("=========================")
a = torch.zeros(2,2)
print(a)
print(a.type)


print("=========================")
a = torch.ones(2,2)
print(a)
print(a.type)

print("=========================")
a = torch.zeros_like( a)
print(a)
print(a.type)


print("=========================")
a = torch.ones_like(a)
print(a)
print(a.type)


"""随机"""
print("===========随机==============")
a = torch.rand(2,2)
print(a)
print(a.type)

a = torch.normal( mean=0.0, std=torch.rand(5))
print(a)
print(a.type)

print("=========================")
# uniform_() 是 PyTorch 中的一个就地(inplace)随机采样方法，用于将张量中的元素替换为从均匀分布中采样的随机值。
a = torch.Tensor(2,2).uniform_(-1,1) #
print(a)
print(a.type)


"""序列"""
a = torch.arange(0,10)
print(a)
print(a.type)

print("=========================")
# range 被弃用，请使用 arange() 代替
a = torch.range(0,10)
print(a)
print(a.type)

print("=========================")
a = torch.range(0,10,2)
print(a)
print(a.type)

print("=========================")
a = torch.linspace(0,10,5)
print(a)
print(a.type)


