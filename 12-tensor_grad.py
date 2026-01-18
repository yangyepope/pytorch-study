# 定义函数
import torch

x = torch.tensor(10.0)
y = torch.tensor([[3.0]])
print(x)
print(y)


# 初始化参数
w = torch.rand(1, 1, requires_grad=True)
print(w)
b = torch.rand(1, 1, requires_grad=True)
print(b)


# 前向传播，得到输出
z = w * x + b
print(z)
# tensor([[9.3561]], grad_fn=<AddBackward0>)

print(x.is_leaf)
print(w.is_leaf)
print(b.is_leaf)
print(y.is_leaf)
print(z.is_leaf)

# 通过叶子结点计算梯度，非叶子阶段不计算梯度，除非定义return_grad=True，否则无法计算梯度


# 设置损失函数
loss = torch.nn.MSELoss()
loss_value = loss(z, y)
print(loss_value)


# 反向传播
loss_value.backward()  # 计算梯度会对loss进行判断，如果loss是一个标量，则计算梯度
print(w.grad)
print(b.grad)
print(x.grad)
print(y.grad)
print(w.requires_grad)
