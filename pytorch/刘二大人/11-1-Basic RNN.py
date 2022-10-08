from cProfile import label
import torch

def cell():
    input_size=8
    hidden_size=16
    cell=torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)
    hidden=cell(input,hidden)   #!input=(batch,input_size)  hidden=(batch,hidden_size)

#warn 现有一个样本，seqLen为3，input_size=4\hiddensize=2
#warn dataset.shape=(seqLen,batch_size,input_size)
def RNNcell_to_Net():
    batch_size=1
    seq_len=3
    input_size=4
    hidden_size=2
    cell=torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)

    dataset=torch.randn(seq_len,batch_size,input_size)
    hidden=torch.zeros(batch_size,hidden_size)

    for idx,input in enumerate(dataset):
        print('='*20,idx,'='*20)
        print('Input size:',input.shape)

        hidden=cell(input,hidden)

        print('outputs size:',hidden.shape)
        print(hidden)

def RNN_Net():
    batch_size=1
    seq_len=3
    input_size=4
    hidden_size=2
    num_layers=1   #warn 隐层的层数

    cell=torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
    
    inputs=torch.randn(seq_len,batch_size,input_size)
    hidden=torch.zeros(num_layers,batch_size,hidden_size)

    outputs,hidden=cell(inputs,hidden)

    print('Output Size:',outputs.shape)
    print('Output:',outputs.data)
    print('Hidden Size:',hidden.shape)
    print('Hidden:',hidden.data)

RNN_Net()





