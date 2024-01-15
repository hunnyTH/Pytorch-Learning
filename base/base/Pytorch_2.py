import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1,6,5)   # 卷积层1
        self.conv2 = nn.Conv2d(6,16,5)  # 卷积层2
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5,120)  # 全连接层1；5*5 from image dimension    
        self.fc2 = nn.Linear(120,84)    #全连接层2
        self.fc3 = nn.Linear(84,10)     #全连接层3（输出层）
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))   # 池化层1
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)),2)   # 池化层2
        x = torch.flatten(x,1) # flatten all dimensions except the batch dimension
        
        x = F.relu(self.fc1(x)) # 全连接层1计算
        x = F.relu(self.fc2(x)) # 全连接层2计算
        x = self.fc3(x)         # 全连接层2（输出层）计算
        return x    # 返回结果
    
net = Net() # 定义网络
print(net)  # 查看网络

params = list(net.parameters())
print(len(params))
for v in range(len(params)):
    print(params[v].size()) # weight

#随即输入32*32
input = torch.randn(1,1,32,32)
out = net(input)
print(input)
print(out)

#使用随机梯度将所有参数和反向传播的梯度缓冲区归零：
net.zero_grad()
out.backward(torch.randn(1, 10))

"""损失函数"""

output = net(input)
target = torch.randn(10)
target = target.view(1,-1)  #张量平铺
critersion = nn.MSELoss()
loss = critersion(output, target)
print(loss)

print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

"""反向传播"""
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

"""更新权重"""
# create your optimizer
optimizer = optim.SGD(net.parameters(),lr=0.01)

# in your training loop:
for v in range(100):
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = critersion(output,target)
    loss.backward()
    optimizer.step()    # Does the update

    print(net.conv1.bias.grad)
    print(loss)
