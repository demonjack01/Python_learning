from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt



batch_size=64
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])   #step 归一化
train_dataset=datasets.MNIST(root='../dataset/minist',train=True,transform=transform,download=True)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_dataset=datasets.MNIST(root='../dataset/minist',train=False,transform=transform,download=True)
test_loader=DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

class InceptionA(torch.nn.Module):
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        #step 1x1分支
        self.branch1x1=nn.Conv2d(in_channels,16,kernel_size=1)
        #step 5x5分支
        self.branch5x5_1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2=nn.Conv2d(16,24,kernel_size=5,padding=2)
        #step 3x3分支
        self.branch3x3_1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2=nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3=nn.Conv2d(24,24,kernel_size=3,padding=1)
        #step pooling分支
        self.branch_pool=nn.Conv2d(in_channels,24,kernel_size=1)
    def forward(self,x):
        #step
        branch_1x1=self.branch1x1(x)
        #step
        branch_5x5=self.branch5x5_1(x)
        branch_5x5=self.branch5x5_2(branch_5x5)
        #step
        branch_3x3=self.branch3x3_1(x)
        branch_3x3=self.branch3x3_2(branch_3x3)
        branch_3x3=self.branch3x3_3(branch_3x3)
        #step
        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)

        outputs=[branch_1x1,branch_5x5,branch_3x3,branch_pool]
        return torch.cat(outputs,dim=1)                          #!!沿channel拼接

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(88,20,kernel_size=5)

        self.incep1=InceptionA(in_channels=10)
        self.incep2=InceptionA(in_channels=20)

        self.mp=nn.MaxPool2d(2)
        self.fc=nn.Linear(1408,10)

    def forward(self,x):
        in_size=x.size(0)
        x=F.relu(self.mp(self.conv1(x)))
        x=self.incep1(x)
        x=F.relu(self.mp(self.conv2(x)))
        x=self.incep2(x)
        x=x.view(in_size,-1)
        x=self.fc(x)
        return x

model=Net()
device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model.to(device)

criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        inputs,target=inputs.to(device),target.to(device)
        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx%300==299:
            print('[%d %5d] loss:%.3f'%(epoch+1,batch_idx+1,running_loss/2000))
            running_loss=0.0

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            inputs,target=data
            inputs,target=inputs.to(device),target.to(device)
            outputs=model(inputs)
            _,predicted=torch.max(outputs.data,dim=1)
            total+=target.size(0)
            correct+=(predicted==target).sum().item()
    print('Accuracy on test set:%d %%d [%d %d]' % (100*correct/total,correct,total))
    return correct/total

if __name__ == '__main__':
    epoch_list = []
    acc_list = []
    
    for epoch in range(10):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)
    
    plt.plot(epoch_list,acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
