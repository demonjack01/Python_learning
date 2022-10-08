from turtle import forward
import torch
import numpy as np

xy=np.loadtxt('diabetes.csv.gz',delimiter=",",dtype=np.float32)
x_data=torch.from_numpy(xy[:,:-1])
y_data=torch.from_numpy(xy[:,[-1]])

class Multiple_Model(torch.nn.Module):
    def __init__(self):
        super(Multiple_Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3.torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()    

    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x


