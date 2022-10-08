# import torchvision
# train_set=torchvision.datasets.MNIST(root='../dataset/minist',train=True,download=True)
# test_set=torchvision.datasets.MNIST(root='../dataset/minist',train=False,download=True)
## CIFAR10数据集

from random import random
from turtle import forward
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred

x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[0],[0],[1]])

model=LogisticRegressionModel()
criterion=torch.nn.BCELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

loss_list=[]
epoch_list=[]
for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())

    criterion.zero_grad()
    loss.backward()
    optimizer.step()
    
    print('progress:', epoch, loss.item())
    loss_list.append(loss.item())
    epoch_list.append(epoch)

