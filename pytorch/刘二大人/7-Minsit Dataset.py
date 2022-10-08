from importlib import import_module
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

train_dataset=datasets.MNIST(root='../dataset/minist',train=True,transform=transforms.ToTensor(),download=True)
test_datatest=datasets.MNIST(root='../dataset/minist',train=False,transform=transforms.ToTensor(),download=True)

train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(dataset=test_datatest,batch_size=32,shuffle=False)

