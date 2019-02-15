import torch
from torch.autograd import Variable


a = Variable(torch.Tensor([1]))

b = Variable(torch.Tensor([2]))


# in place add_
# a.data.add_(b)
#
# print(a)  tensor([3])  a + 1*b

a.data.add_(2, b)


print(a) # a + 2*b
