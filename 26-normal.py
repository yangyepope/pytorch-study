import torch
import torch.nn as nn

linear = nn.Linear(5, 2)

# 实验 A：std=2.0 (你现在的设置，像个狂暴的菜鸟)
nn.init.normal_(linear.weight, mean=0.0, std=2.0)
print("std=2.0 的权重:\n", linear.weight.data)

# 实验 B：std=0.01 (像个胆小的谨慎者)
nn.init.normal_(linear.weight, mean=0.0, std=0.01)
print("\nstd=0.01 的权重:\n", linear.weight.data)