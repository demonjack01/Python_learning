from cProfile import label
import imp
from turtle import forward
import torch
import numpy as np
from sklearn.datasets import load_diabetes
from torch.utils.data import Dataset         #! Dataset无法实例化
from torch.utils.data import DataLoader


class DiabetesDataset(Dataset):
    def __init__(self,filepath):              #! 要加路径的话为__init__(self,filepath)
        xy=np.loadtxt(filepath,delimiter=",",dtype=np.float32)           #! np.loadtxt(filepath,delimiters,dtype)
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,:-1])
        self.y_data=torch.from_numpy(xy[:,[-1]])
    def __getitem__(self, index):     #!! 支持索引操作
        return self.x_data[index], self.y_data[index]    
    def __len__(self):
        return self.len


dataset=DiabetesDataset('diabetes.csv.gz')
train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)


class Multiple_Model(torch.nn.Module):
    def __init__(self):
        super(Multiple_Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()    

    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x

model=Multiple_Model()
criterion=torch.nn.BCELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)


if __name__=='__main__':    #!! 设了num_workers
    train_epochs=100
    for epoch in range(train_epochs):
        for i ,data in enumerate(train_loader,0):       #!可以获得训练次数
            inputs, labels=data
            y_pred=model(inputs)
            loss=criterion(y_pred,labels)
            print(epoch,i,loss.item())

            criterion.zero_grad()
            loss.backward()
            optimizer.step()
    

