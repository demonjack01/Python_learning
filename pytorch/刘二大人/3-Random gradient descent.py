from matplotlib import pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w=0

def forward(x):
    return w*x

def loss(xs,ys):
    y_pred=forward(xs)
    return (y_pred-ys)**2

def gradient(xs,ys):
    return 2*xs*(xs*w-ys)


epoch_list = []
loss_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        grad=gradient(x,y)
        w=w-0.01*grad
        print("\tgrad:", x, y,grad)
        l = loss(x,y)
    print("progress:",epoch,"w=",w,"loss=",l)
    epoch_list.append(epoch)
    loss_list.append(l)

print('predict (after training)', 4, forward(4))


