import torch


# torch.randint(low, high, size)
x = torch.randint(1, 10, (3, 2, 4))
print(x)
print(id(x))

# x = x + 10
# 使用加等于方法共享内存
x += 10
print(x)
print(id(x))

x.add_(10)
print(
     x
)
print(id(x))

x = torch.randint(1, 10, (3, 2, 4))
y = torch.randint(1, 10, (3, 4, 2))

print(x.shape)
print(id(x))

# x = x @ y
x @= y

print(x.shape)
# 形状已经改变导致ids不同
print(id(x))


print(x.size())


#  节省内存的方案
# x[:] = x @ y
# print(x.size())
# print(id(x))


X = torch.randint(1, 10, (3, 2, 4))
Y = torch.randint(1, 10, (3, 4, 1))
print(X.shape)
print(id(X))
print(X)
print(Y)
print(X @ Y)
# X = X @ Y
# X @= Y
X[:] = X @ Y
print(X.shape)
print(X)
print(id(X))



