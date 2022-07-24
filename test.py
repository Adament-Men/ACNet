import torch.nn as nn
import torch

m = nn.AdaptiveMaxPool3d((1, 5, 5))
input = torch.randn(1, 3, 5, 5)
output = m(input)
print(output.shape)

pass
# target output size of 7x7 (square)
# m = nn.AdaptiveAvgPool2d(7)
# input = torch.randn(1, 64, 10, 9)
# output = m(input)
# print(output.shape)
#
# # target output size of 10x7
# m = nn.AdaptiveMaxPool2d((None, 7))
# input = torch.randn(1, 64, 10, 9)
# output = m(input)
# print(output.shape)
