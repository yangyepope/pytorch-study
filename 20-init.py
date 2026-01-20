import torch
import torch.nn as nn


# 定义一个全连接层
linear = nn.Linear(5, 2) # 5个输入，2个输出


# 1. 常数初始化
nn.init.zeros_(linear.weight)
print(linear.weight)

#output


"""
Parameter containing:
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]], requires_grad=True)
"""

# 偏置初始化
nn.init.ones_(linear.bias)
print(linear.bias)

#output
"""
Parameter containing:
tensor([1., 1.], requires_grad=True)

"""


nn.init.constant_(linear.weight, 10)
print(linear.weight)

#output
"""
Parameter containing:
tensor([[10., 10., 10., 10., 10.],
        [10., 10., 10., 10., 10.]], requires_grad=True)
"""

# 2. 秩初始化
nn.init.eye_(linear.weight)
print(linear.weight)

# output

"""
Parameter containing:
tensor([[1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.]], requires_grad=True)

"""

# 3. 正态分别
nn.init.normal_(linear.weight, mean=0.0, std=2.0)
print(linear.weight)


# 4. 均匀分布初始化
nn.init.uniform_(linear.weight, a=-3.0, b=3.0)
print(linear.weight)


# 5. Xavier初始化
nn.init.xavier_normal_(linear.weight)
print(linear.weight)


nn.init.xavier_uniform_(linear.weight)
print(linear.weight)

# 6. He初始化或Kaiming初始化
nn.init.kaiming_normal_(linear.weight)
print(linear.weight)