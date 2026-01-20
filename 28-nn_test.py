import torch
import torch.nn as nn


# 自定义神经网络类
class SimpleNN(nn.Module):
    def __init__(self, device='cpu'):
        super(SimpleNN, self).__init__()
        # 定义3个线性层

        # 定义第一个线性层，输入特征数为3，输出特征数为4
        self.linear1 = nn.Linear(3, 4,device=device)
        nn.init.xavier_normal_(self.linear1.weight)

        # 创建第二个线性层，输入特征数为4，输出特征数为4
        self.linear2 = nn.Linear(4, 4,device= device)
        nn.init.kaiming_normal_(self.linear2.weight)

        # 创建第三个线性层，输入特征数为4，输出特征数为2
        self.linear3 = nn.Linear(4, 2,device= device)

    def forward(self, x):
        # 前向传播

        # 第一个线性层,使用tanh激活函数
        x = self.linear1(x)
        # x = nn.functional.tanh(x)
        x = torch.tanh(x)

        # 第二个线性层,使用ReLU激活函数
        x = self.linear2(x)
        # x = nn.functional.relu(x)
        x = torch.relu(x)

        # 第三个线性层,使用softmax激活函数
        x = self.linear3(x)
        # x = nn.functional.softmax(x, dim=1)
        x = torch.softmax(x, dim=1)

        return x


# 测试神经网络
if __name__ == "__main__":
    model = SimpleNN()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)

    output = model(x=torch.randn(10, 3))
    print(output)

    # 查看nn参数
    print("==========first layer==============")
    print("weight:", model.linear1.weight)
    print("bias:", model.linear1.bias)

    print("==========second layer==============")
    print("weight:",model.linear2.weight)
    print("bias:",model.linear2.bias)

    print("==========third layer==============")
    print("weight:", model.linear3.weight)
    print("bias:", model.linear3.bias)

    # 调用parameters()方法
    print("=========all parameters=========")
    for param in model.parameters():
        print(param)


    # 查看named_parameters()方法
    print("=========named parameters=========")
    for name, param in model.named_parameters():
        print(name, param)

    # 查看state_dict()方法
    print("=========state_dict=========")
    print(model.state_dict())