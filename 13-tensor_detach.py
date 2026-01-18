import torch

x = torch.tensor(2.0, requires_grad=True)
y = x.detach()
print(x)
print(y)
print(x.requires_grad)
print(y.requires_grad)
print(id(x))
print(id(y))

# 底层数据源相同共享
print(x.untyped_storage().data_ptr())
print(y.untyped_storage().data_ptr())

# 验证底层数据源
# print(x)
# y.zero_()
# print(x)

z1 = x ** 2
z2 = y ** 2
print(z1)
print(z2)

z1.sum().backward()  # sum函数可以将张量转换成标量
# z2.sum().backward()

# print(z1.grad)
# print(z2.grad)
print(x.grad)

x = torch.ones(2, 2, requires_grad=True)
y = x * x
print(x)


"""
<MulBackward0> 表示：
当前这个张量，是通过一个“乘法操作（mul）”得到的，
并且它带有反向传播能力（Backward）

tensor([[1., 1.],
        [1., 1.]], grad_fn=<MulBackward0>)

"""
print(y)


u = y.detach()  # u是有detach出来的张量，不会进入到计算图的 中，将u当做一个常数处理
print("打印 u")
print( u)

# z = x * x * x
# print("z:",z)
#
# # 反向传播
# z.sum().backward()
# print(x.grad)
#
# """
# tensor([[3., 3.],
#         [3., 3.]])
# """


# 将u当做一个常数处理
z = u * x
z.sum().backward()
print(x.grad)


"""
tensor([[1., 1.],
        [1., 1.]])
tensor([[1., 1.],
        [1., 1.]])

"""