import torch
import numpy as np

input_size=4
hidden_size=4
batch_size=1
idx2char=['e','h','l','o']   #! 字典标签的顺序随便，因为最后要与字典标签对应
x_data=[1,0,2,2,3]
y_data=[3,1,2,3,2]
one_hot_lookup=np.eye(4,4)
x_one_hot=np.array([one_hot_lookup[x] for x in x_data])
print(x_one_hot)
inputs=torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
labels=torch.LongTensor(y_data).view(-1,1)

class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(Model,self).__init__()
        self.batch_size=batch_size
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.rnncell=torch.nn.RNNCell(input_size=self.input_size,hidden_size=self.hidden_size)
    
    def forward(self,input,hidden):
        hidden=self.rnncell(input,hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)

net=Model(input_size,hidden_size,batch_size)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.1)

for epoch in range(15):
    loss=0
    optimizer.zero_grad()
    hidden=net.init_hidden()
    print('Predicted String:',end='')
    
    for input,label in zip(inputs,labels):
        hidden=net(input,hidden)
        loss+=criterion(hidden,label)
        _,idx=hidden.max(dim=1)
        print(idx2char[idx.item()],end='')
    
    loss.backward()
    optimizer.step()
    print(',Epoch [%d/15] loss=%.4f' % (epoch+1,loss.item()))

