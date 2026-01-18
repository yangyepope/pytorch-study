import torch

# 张量转标量

tensor1 = torch.tensor(10)
print(tensor1.item())

torch_tensor = torch.tensor([1, 2, 3])
print(torch_tensor.sum())

"""

Traceback (most recent call last):
  File "D:\1-application\python_project\pytorch-study\10-tensor_scalar.py", line 10, in <module>
    print(torch_tensor.item() )
          ^^^^^^^^^^^^^^^^^^^
RuntimeError: a Tensor with 3 elements cannot be converted to Scalar

"""
# 只能提取一个元素
# print(torch_tensor.item() )

tensor2 = torch.tensor([[10]])
print(tensor2)
print(tensor2.item())
