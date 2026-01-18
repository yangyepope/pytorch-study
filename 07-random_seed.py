import random

print(random.randint(1, 100))



# add  seed

import random

random.seed(42)
print(random.randint(1, 100))


# add seed in torch


import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

"""
应用场景：
1. 数据打乱
dataloader = DataLoader(dataset, shuffle=True)


没种子：每次打乱顺序不同

有种子：打乱顺序固定

2. 数据增强

transforms.RandomHorizontalFlip()
transforms.RandomRotation()
不设种子：每次增强方式不同

设种子：增强方式可复现

3. 模型初始化

model = ResNet()
权重初始化本身就是随机的

设种子后 → 初始化固定


4. Dropout
nn.Dropout(0.5)

Dropout 是随机丢神经元

种子决定丢哪些
"""


# 随机重新排列
randperm = torch.randperm(10)
print(randperm)

# 查看和制定随机种子
print(torch.initial_seed())


# 手动更改随机种子
torch.manual_seed(41)
print(torch.initial_seed())