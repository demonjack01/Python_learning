from pickletools import optimize
import torch
import torch.nn as nn

num_class=4
input_size=4
hidden_size=8
embedding_size=10
num_layers=2
batch_size=1
seq_len=5

idx2char=['e','h','l','0']
x_data=[[1,0,2,2,3]]    #! (batch,seq_len)
y_data=[3,1,2,3,2]      #! (batch_size*seq_len)
inputs=torch.LongTensor(x_data)
labels=torch.LongTensor(y_data)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.emb=nn.Embedding(input_size,embedding_size)
        self.rnn=nn.RNN(input_size=embedding_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.fc=torch.nn.Linear(hidden_size,num_class)

    def forward(self,x):
        hidden=torch.zeros(num_layers,x.size(0),hidden_size)
        x=self.emb(x)
        x,_=self.rnn(x,hidden)
        x=self.fc(x)
        return x.view(-1,num_class)

net=Model()
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


