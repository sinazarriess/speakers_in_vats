import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F


a = torch.randn(10, 3)
b = torch.randn(10, 3)
c = torch.randn(10, 5, 3)

print(a)
print(b)
score = torch.sum(torch.mul(a, b), dim=1)
print(score)
score = torch.clamp(score, max=10, min=-10)
print(score)
score = -F.logsigmoid(score)
print("final",score)


print("unsqueeze",a.unsqueeze(2))
neg_score = torch.bmm(c, a.unsqueeze(2)).squeeze()
print(neg_score)

neg_score = torch.clamp(neg_score, max=10, min=-10)
neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
print("final",neg_score)
