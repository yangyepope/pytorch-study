import torch


# add
print("===================加法运算===================")
a = torch.rand(2,3)
b = torch.rand(2,3)

print(a)
print(b)

print(a+b)
print(torch.add(a,b))
print(a.add(b))
print(a,"======before========")
print(a.add_(b))
print(a,"======after========")




# sub
print("===================减法运算===================")
print(a-b)
print(torch.sub(a,b))
print(a.sub(b))
print(a,"======before========")
print(a.sub_(b))
print(a,"======after========")


# multiplication 哈达玛积
print("===================乘法运算===================")
print(a*b)
print(torch.mul(a,b))
print(a.mul(b))
print(a,"======before========")
print(a.mul_(b))
print(a,"======after========")


# division
print("===================除法运算===================")
print(a/b)
print(torch.div(a,b))
print(a.div(b))
print(a,"======before========")
print(a.div_(b))
print(a,"======after========")

# 矩阵乘法
print("===================矩阵乘法===================")
a = torch.ones(2,1)
b = torch.ones(1,2)
print(a@ b)
print(a.mm(b))
print(torch.mm(a,b))
print(torch.matmul(a,b))
print(torch.mm(a,b))


print("===================矩阵乘法===================")
a = torch.ones(1,2)
b = torch.ones(2,1)
print(a@ b)
print(a.mm(b))
print(torch.mm(a,b))
print(torch.matmul(a,b))
print(torch.mm(a,b))


# 高维矩阵乘法

a = torch.ones(1,2,3,4)
b = torch.ones(1,2,4,3)
# print(a@ b)
print(a.matmul(b).shape)

print("===================矩阵乘法?===================")
a = torch.ones(1,2,4,3)
b = torch.ones(1,2,3,4)
print(a.matmul(b))


# power 指数运算
print("===================指数运算===================")
a = torch.tensor([1,2])
print(a.pow(3))
print(torch.pow(a,3))
print(a**3)