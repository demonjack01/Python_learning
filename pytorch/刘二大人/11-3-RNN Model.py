import torch
import torch.nn as nn
import numpy as np

input_size=4
hidden_size=4
batch_size=1
num_layers=1
seq_len=5
idx2char=['e','h','l','o']   #! 字典标签的顺序随便，因为最后要与字典标签对应
x_data=[1,0,2,2,3]
y_data=[3,1,2,3,2]
one_hot_lookup=np.eye(4,4)
x_one_hot=np.array([one_hot_lookup[x] for x in x_data])
print(x_one_hot)
inputs=torch.Tensor(x_one_hot).view(seq_len,batch_size,input_size)
labels=torch.LongTensor(y_data)


class Model(nn.Module):
    def __init__(self,batch_size,input_size,hidden_size,num_layers):
        super(Model,self).__init__()
        self.batch_size=batch_size
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.rnn=nn.RNN(num_layers=self.num_layers,input_size=self.input_size,hidden_size=self.hidden_size)

    def forward(self,input):
        hidden=torch.zeros(self.num_layers,self.batch_size,self.hidden_size)  #!初始化也可以在forward里
        out,_=self.rnn(input,hidden)     #! hidden层直接不要
        return out.view(-1,self.hidden_size)

net=Model(batch_size,input_size,hidden_size,num_layers)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.1)

for epoch in range(15):
    optimizer.zero_grad()
    outputs=net(inputs)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()

    _,idx=outputs.max(dim=1)
    idx=idx.data.numpy()

    print('Predicted: ',''.join([idx2char[x] for x in idx]),end='')
    print(',Epoch [%d/15] loss=%.4f' % (epoch+1,loss.item()))
