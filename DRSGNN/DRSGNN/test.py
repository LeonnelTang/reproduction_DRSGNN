import torch

print(torch.__version__)
print(torch.cuda.is_available())
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())

x = torch.rand(5, 3)
print(x)