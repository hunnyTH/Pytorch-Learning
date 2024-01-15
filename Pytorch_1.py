import torch
import numpy as np

""" 张量初始化"""

#直接从数据创建张量
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

#从Numpy数组创建
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

#从另一个张量(除非明确覆盖，否则新张量保留参数张量的属性（形状、数据类型）)
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data,dtype=torch.float)# overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

#使用随机值或常量值
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


"""张量属性"""
#形状，数据类型，存储设备
tensor = torch.rand(3,4)
print(f"Shape of tensor:{tensor.shape}")
print(f"Dtattype of tensor:{tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

"""张量操作"""
tensor = torch.ones(4,4)
tensor[:,1] = 0
print(tensor)

#连接张量
t1 = torch.cat([tensor,tensor,tensor],dim=1)    #dim = 1, 横向连接
print(t1)
t2 = torch.cat([tensor,tensor,tensor],dim=0)    #dim = 0, 纵向连接
print(t2)

#乘法张量

#位乘
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

#矩阵相乘
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

#就地操作："_"后缀，
print(tensor,"\n")
tensor.add_(5)
print(tensor)

"""使用Numpy桥接"""
#CP 和 NumPy 数组上的张量可以共享它们的底层内存位置，改变一个将改变另一个。
t = torch.ones(5)
n = t.numpy()
print(f"t:{t}")
print(f"n:{n}")
#张量的变化反映在 NumPy 数组中。
t.add_(1)
print(f"t:{t}")
print(f"n:{n}")

n = np.ones(5)
t = torch.from_numpy(n)
#NumPy 数组中的变化反映在张量中。
np.add(n,1,out=n)
print(f"t:{t}")
print(f"n:{n}")
