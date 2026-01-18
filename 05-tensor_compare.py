
import torch



print("===========1.比较运算==================")
a = torch.rand(2,2)
b = torch.rand(2,2)


print(a)
print(b)

print(torch.eq(a,b))
print(torch.equal(a,b))


print(torch.ge(a,b))
print(torch.gt(a,b))
print(torch.le(a,b))
print(torch.lt(a,b))
print(torch.ne(a,b))



a = torch.ones(2,2)
b = torch.ones(2,2)


print(a)
print(b)

print(torch.eq(a,b))
print(torch.equal(a,b))


print(torch.ge(a,b))
print(torch.gt(a,b))
print(torch.le(a,b))
print(torch.lt(a,b))
print(torch.ne(a,b))



print("===========2.排序运算==================")

a = torch.tensor([1,4,4,3,5])
print(torch.sort(a,dim=0,descending=False))



a = torch.tensor([[1,4,4],[3,5,6]])
print(torch.sort(a,descending=True))


a = torch.tensor([[1,4,4],[3,5,6]])
print(torch.sort(a,dim=0,descending=True))
print(a.shape)
print(a.size())


# topk
a = torch.tensor([[2,4,3,1,5],[2,3,5,1,4]])
print(torch.topk(a,k=2,dim=0))