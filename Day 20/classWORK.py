import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

train_data = datasets.FashionMNIST(root = 'data', train=True, download = True, transform = ToTensor())

test_data = datasets.FashionMNIST(root = 'data', train=False, download = True, transform = ToTensor())
