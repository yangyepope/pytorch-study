import torch
from torch import nn, optim


# 定义模型
class Net(nn.Module):
    # 初始化
    def __init__(self):
        # 继承父类初始化
        super(Net, self).__init__()
        # 定义一个全连接层
        self.linear = nn.Linear(5, 3)

        # 定义权重
        # self.linear.weight.data = torch.tensor(
        #     [
        #         [0.1,0.2,0.3],
        #         [0.2,0.3,0.4],
        #         [0.3,0.4,0.5],
        #         [0.4,0.5,0.6],
        #         [0.5,0.6,0.7]
        #
        #
        #     ]).T

        self.linear.weight.data = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.3, 0.4, 0.5, 0.6],
                [0.3, 0.4, 0.5, 0.6, 0.7],
            ])

        # 定义偏置
        self.linear.bias.data = torch.tensor([0.1, 0.2, 0.3]) # 建议使用这种方式初始化偏置
        # self.linear.bias.data = torch.tensor([[0.1, 0.2, 0.3]])

    # 前向传播
    def forward(self, x):
        x = self.linear(x)
        return x

# 创建模型
model = Net()
print(model)

# 定义输入
input = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],dtype=torch.float32)
output = model(input)
print(output)

# 定义目标
target = torch.tensor([[0.5, 0.6, 0.7], [0.6, 0.7, 0.8]], dtype=torch.float32)

# 定义损失函数
criterion = nn.MSELoss()
loss_value = criterion(output, target)

# 反向传播
loss_value.backward()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 更新参数
optimizer.step()
optimizer.zero_grad()

for parameter in model.state_dict():
    print(parameter, model.state_dict()[parameter])


