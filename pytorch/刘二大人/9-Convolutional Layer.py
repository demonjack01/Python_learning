import torch

def Convolutional_Layer():
    in_channels,out_channels=5,10   #step 定义输入、输出通道
    width,height=10,10
    kernel_size=3  #step 卷积核大小
    batch_size=1

    input=torch.randn(batch_size,in_channels,width,height)
    conv_layer=torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size)
    output=conv_layer(input)

    print(input.shape)
    print(output.shape)
    print(conv_layer.weight.shape)

def Convolutional_layer_padding():
    input=[3,4,6,5,7,2,4,6,8,2,1,6,7,8,4,9,7,4,6,2,3,7,5,4,1]
    input=torch.Tensor(input).view(1,1,5,5)
    conv_layer=torch.nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)
    kernal=torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3,)
    conv_layer.weight.data=kernal.data
    output=conv_layer(input)
    print(output)

def Convolutional_layer_stride():
    input=[3,4,6,5,7,2,4,6,8,2,1,6,7,8,4,9,7,4,6,2,3,7,5,4,1]
    input=torch.Tensor(input).view(1,1,5,5)
    conv_layer=torch.nn.Conv2d(1,1,kernel_size=3,stride=2,bias=False)
    kernal=torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3,)
    conv_layer.weight.data=kernal.data
    output=conv_layer(input)
    print(output)

def MaxPooling_Layer():
    input=torch.randn(1,1,4,4)
    maxpooling_layer=torch.nn.MaxPool2d(kernel_size=2)   #! 默认步长会根据kernal_size变化


#step1 基本卷积层
#Convolutional_Layer()
#step2 外圈
#Convolutional_layer_padding()
#step3 步长
#Convolutional_layer_stride()
#step4 池化层（通道数不变）
#MaxPooling_Layer()

