import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


train_data = datasets.FashionMNIST(root='data', train=True,
                                   download=False, transform=ToTensor())

test_data = datasets.FashionMNIST(root='data', train=False,
                                   download=False, transform=ToTensor())

print(len(train_data))
print(len(train_data[0]))
print(train_data.data.shape)

plt.figure()
plt.imshow(train_data.data[0])
plt.show()