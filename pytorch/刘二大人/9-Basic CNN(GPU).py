from turtle import forward
import torch
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

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.pooling=torch.nn.MaxPool2d(2)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.linear=torch.nn.Linear(320,10)

    def forward(self,x):
        batch_size=x.size(0)
        x=F.relu(self.pooling(self.conv1(x)))
        x=F.relu(self.pooling(self.conv2(x)))
        x=x.view(batch_size,-1)
        x=self.linear(x)
        return x
    
model=CNN()
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

