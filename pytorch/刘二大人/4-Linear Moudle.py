import torch

x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred  

model=LinearModel()
criterion=torch.nn.MSELoss(reduction='sum')              #!! MSEloss函数
optimizer=torch.optim.SGD(model.parameters(), lr=0.01)   #!! SGD函数

for epoch in range(100):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()     #! zero_grad()清零
    loss.backward()           #! backward()反向传播
    optimizer.step()          #! step()更新

print('w=', model.linear.weight.item())   #! .weight获取w
print('b=',model.linear.bias.item())      #! .bias获取b

x_test=torch.Tensor([[4.0],[5.0]])
y_test=model(x_test)
print('y_pred=',y_test.data)

torch.save(model.state_dict(),'model.pkl')
model.load_state_dict(torch.load('model.pkl'))
