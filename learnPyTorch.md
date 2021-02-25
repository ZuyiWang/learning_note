# PyTorch学习笔记 
# 1. [英文官方文档](https://pytorch.org/tutorials/)
## Chapter 1 Learning PyTorch
---
### Deep Learning with PyTorch: A 60 Minute Blitz
---
+ What's PyTorch?
   + 它是一个基于python的科学计算包，可以实现两个目的:代替NumPy从而可以使用gpu的力量; 可以帮助实现神经网络的自动微分库;
#### Tensors
---
   1. **张量的初始化**
      ```python
      import torch
      import numpy
      ```
   + 从数据转化
      ```python
      data = [[1, 2],[3, 4]]
      x_data = torch.tensor(data) 
      ```
   + 由Numpy Array创建张量
      ```python
      np_array = np.array(data)
      x_np = torch.from_numpy(np_array)    
      ```      
   + 基于已经存在的张量创建新张量, 相关数据类型等属性会复用输入张量的, 除非用户重新定义
      ```python
      x = x.new_ones(5, 3, dtype=torch.double)
      print(x)
      
      # tensor([[1., 1., 1.],
      #         [1., 1., 1.],
      #         [1., 1., 1.],
      #         [1., 1., 1.],
      #         [1., 1., 1.]], dtype=torch.float64)
      
      x = torch.randn_like(x, dtype=torch.float) # 输出与输入的size相同
      print(x)
      
      # tensor([[-0.7461,  2.3056, -1.3214],
      #         [ 0.2807, -0.4332,  0.7302],
      #         [-0.7813,  1.1492, -1.5275],
      #         [-0.2631, -0.5270, -1.9110],
      #         [-0.8535, -0.3139, -0.4333]])
      ```
   + 创建未初始化的矩阵, 在没有初始化之前，其初始值为内存中的值
      ```python
         x = torch.empty(5, 3)
         print(x)
         
         # tensor([[ 7.1441e+31,  6.9987e+22,  7.8675e+34],
         #         [ 4.7418e+30,  5.9663e-02,  7.0374e+22],
         #         [ 1.6195e-37,  0.0000e+00,  1.6195e-37],
         #         [ 0.0000e+00,  1.8077e-43,  0.0000e+00],
         #         [-5.3705e-17,  4.5717e-41, -5.3705e-17]])
      ```
   + 创建随机初始化的矩阵
      ```python
      x = torch.rand(5, 3)
      print(x)
      
      # tensor([[0.8446, 0.2635, 0.9593],
      #         [0.1011, 0.9376, 0.1164],
      #         [0.0733, 0.0423, 0.1420],
      #         [0.8750, 0.1228, 0.7428],
      #         [0.7618, 0.8400, 0.5244]])
      ```
   + 创建零矩阵, 数值类型为long
      ```python
      x = torch.zeros(5, 3, dtype=torch.long)
      print(x)
      
      # tensor([[0, 0, 0],
      #         [0, 0, 0],
      #         [0, 0, 0],
      #         [0, 0, 0],
      #         [0, 0, 0]])
      ```
   + 直接由数据创建张量
      ```python
      x = torch.tensor([5.5, 3])
      print(x)
      
      # tensor([5.5000, 3.0000])
      ```
   2. **张量的属性**
   + 张量的shape dtype device
      ```python
      tensor = torch.rand(3,4)

      print(f"Shape of tensor: {tensor.shape}")
      print(f"Datatype of tensor: {tensor.dtype}")
      print(f"Device tensor is stored on: {tensor.device}")

      # Shape of tensor: torch.Size([3, 4])
      # Datatype of tensor: torch.float32
      # Device tensor is stored on: cpu
      ```
   + 获取张量大小, 返回值为元组, 支持元组操作
      ```python
      print(x.size())
      # torch.Size([5, 3]) 
      height, width = x.size()
      ```
   3. **张量的操作**
      ```python
      # We move our tensor to the GPU if available
      if torch.cuda.is_available():
         tensor = tensor.to('cuda')
      ```
   + 标准的索引与切片
      ```python
      tensor = torch.ones(4, 4)
      tensor[:,1] = 0
      print(tensor)

      # tensor([[1., 0., 1., 1.],
      #   [1., 0., 1., 1.],
      #   [1., 0., 1., 1.],
      #   [1., 0., 1., 1.]])
      ```
   + 张量的连接 *torch.cat, torch.stack*
      ```python
      t1 = torch.cat([tensor, tensor, tensor], dim=1)
      print(t1)
      
      # tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
      #      [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
      #      [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
      #      [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
      ```
   + 乘法 
      ```python
      # This computes the element-wise product
      print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
      # Alternative syntax:
      print(f"tensor * tensor \n {tensor * tensor}")

      # tensor.mul(tensor)
      #  tensor([[1., 0., 1., 1.],
      #         [1., 0., 1., 1.],
      #         [1., 0., 1., 1.],
      #         [1., 0., 1., 1.]])

      # tensor * tensor
      #  tensor([[1., 0., 1., 1.],
      #         [1., 0., 1., 1.],
      #         [1., 0., 1., 1.],
      #         [1., 0., 1., 1.]])




      # This computes the matrix multiplication between two tensors
      print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
      # Alternative syntax:
      print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

      # tensor.matmul(tensor.T)
      # tensor([[3., 3., 3., 3.],
      #       [3., 3., 3., 3.],
      #       [3., 3., 3., 3.],
      #       [3., 3., 3., 3.]])

      # tensor @ tensor.T
      # tensor([[3., 3., 3., 3.],
      #       [3., 3., 3., 3.],
      #       [3., 3., 3., 3.],
      #       [3., 3., 3., 3.]])
      ```
   + In-place operations: Operations that have a _ suffix are in-place. For example: *x.copy_(y), x.t_(),* will change x.
      ```python
      print(tensor, "\n")
      tensor.add_(5)
      print(tensor)

      # tensor([[1., 0., 1., 1.],
      #       [1., 0., 1., 1.],
      #       [1., 0., 1., 1.],
      #       [1., 0., 1., 1.]])

      # tensor([[6., 5., 6., 6.],
      #       [6., 5., 6., 6.],
      #       [6., 5., 6., 6.],
      #       [6., 5., 6., 6.]])
      ```
   + 加法
      ```python
      y = torch.rand(5, 3)
      print(x+y)
      
      # tensor([[ 0.1525,  3.2603, -1.2916],
      #         [ 0.4684, -0.0993,  1.1413],
      #         [-0.5396,  1.6366, -1.0280],
      #         [ 0.3546, -0.1943, -1.1571],
      #         [-0.3531,  0.6117, -0.3783]])
      
      # 加法二
      print(torch.add(x, y))     
      # tensor([[ 0.1525,  3.2603, -1.2916],
      #         [ 0.4684, -0.0993,  1.1413],
      #         [-0.5396,  1.6366, -1.0280],
      #         [ 0.3546, -0.1943, -1.1571],
      #         [-0.3531,  0.6117, -0.3783]])
      
      # 提供一个输出张量的参数
      torch.add(x, y, out=result)
      print(result)
      # tensor([[ 0.1525,  3.2603, -1.2916],
      #         [ 0.4684, -0.0993,  1.1413],
      #         [-0.5396,  1.6366, -1.0280],
      #         [ 0.3546, -0.1943, -1.1571],
      #         [-0.3531,  0.6117, -0.3783]])      
      ```
      + 修改原变量的加法 ***任何改变张量的操作都是用_后缀。例如:x.copy_(y)， x.t_()都  将改变x。***
      ```python
      y.add_(x)
      print(y)
      # tensor([[ 0.1525,  3.2603, -1.2916],
      #         [ 0.4684, -0.0993,  1.1413],
      #         [-0.5396,  1.6366, -1.0280],
      #         [ 0.3546, -0.1943, -1.1571],
      #         [-0.3531,  0.6117, -0.3783]])              
      ```
   + 可以像Numpy一样对张量使用索引
      ```python
      print(x[:, 1])
      # tensor([ 2.3056, -0.4332,  1.1492, -0.5270, -0.3139])
      ```
   + 改变张量的size, 使用view(), size设置-1代表根据其他维度进行计算
      ```python
      x = torch.rand(4, 4)
      y = x.view(16)
      z = x.view(2, -1)
      print(x.size(), y.size(), z.size())
      
      # torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
      ```
   + 只有一个元素的张量, 可以item()提取元素为数字
      ```python
      x = torch.randn(1)
      print(x)
      print(x.item())
      # tensor([0.7324])
      # 0.7323789000511169
      ```
      
   + 张量范围限定 `torch.clamp(input, max, min, out=None)` input中的元素大于max的赋值为max, 小于min的赋值为min, 其他值保持不变
   + 张量转置 `tensor.t()`
   + [More Operations](https://pytorch.org/docs/stable/torch.html#)
   4. **与Numpy的联系**
   + Torch张量和NumPy数组将共享它们的底层内存位置(如果Torch张量在CPU上)，改变一个将改变另一个。
   + Tensors to Numpy
      ```python
      t = torch.ones(5)
      print(f"t: {t}")
      n = t.numpy()
      print(f"n: {n}")

      # t: tensor([1., 1., 1., 1., 1.])
      # n: [1. 1. 1. 1. 1.]

      # A change in the tensor reflects in the NumPy array.

      t.add_(1)
      print(f"t: {t}")
      print(f"n: {n}")

      # t: tensor([2., 2., 2., 2., 2.])
      # n: [2. 2. 2. 2. 2.]
      ```
   + Numpy to Tensors
      ```python
      n = np.ones(5)
      t = torch.from_numpy(n)
      # Changes in the NumPy array reflects in the tensor.

      np.add(n, 1, out=n)
      print(f"t: {t}")
      print(f"n: {n}")

      # t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
      # n: [2. 2. 2. 2. 2.]
      ```
   + `.to()`改变运行设备
      ```python
      # let us run this cell only if CUDA is available
      # We will use ``torch.device`` objects to move tensors in and out of GPU
      if torch.cuda.is_available():
         device = torch.device("cuda")         # a CUDA device object
         y = torch.ones_like(x, device=device) # directly create a tensor on GPU
         x = x.to(device)                      # or just use strings ``.to("cuda")``
         z = x + y
         print(z)
         print(z.to("cpu", torch.double))      # ``.to`` can also change dtype together!
      ```
##### AutoGrad: 自动微分
---
   1. Tensor
      + 如果将tensor的属性`.requires_grad`设置为`True`，则它将开始跟踪其上的所有操作。当完成计算时，可以调用`.backward()`自动计算所有的梯度。这个张量的梯度会累加到属性`.grad`
      + 要阻止一个tensor追踪历史，可以调用`.detach()`将其从计算历史中分离出来，并防止追踪未来的计算
      + 也可以将代码放入`with torch.no_grad():`的block中, 也不会计算梯度
      + 每一个tensor有属性`.grad_fn`, 引用的创建该张量所涉及的函数
      + **如果张量是一个标量(也就是说，它只包含一个元素数据)，你不需要指定任何参数给`.backward()`，但是如果它有更多的元素，你需要指定一个梯度参数，一个匹配形状的张量**
         ```python
         import torch
         x = torch.ones(2, 2, requires_grad=True)
         print(x)
         # tensor([[1., 1.],
         #         [1., 1.]], requires_grad=True)
         
         y = x + 2
         print(y)     # y was created as a result of an operation, so it has a grad_fn.
         # tensor([[3., 3.],
         #         [3., 3.]], grad_fn=<AddBackward0>)
         
         print(y.grad_fn)
         # <AddBackward0 object at 0x7fcd66c306a0>
         
         z = y * y * 3
         out = z.mean()
         print(z, out)
         # tensor([[27., 27.],
         #         [27., 27.]], grad_fn=<MulBackward0>) 
         # tensor(27., grad_fn=<MeanBackward0>)
         ```
      + **`.requires_grad_()`改变现有张量的requires_grad标志。如果没有给出输入标志，则默认为False**
         ```python
         a = torch.randn(2, 2)
         a = (a*3) / (a-1)
         print(a.requires_grad)
         # False
         
         a.requires_grad_(True)
         print(a.requires_grad)
         # True
         b = (a * a).sum()
         print(b.grad_fn)
         # <SumBackward0 object at 0x7fcd2ed8bb70>
         ```
   2. Gradient 梯度
      + `out`是一个标量, 所以`out.backward()`等价于`out.backward(torch.tensor(1.))`  
         ```python
         out.backward()
         print(x.grad)
         # tensor([[4.5000, 4.5000],
         #         [4.5000, 4.5000]])
         ```
         + You should have got a matrix of `4.5`. Let’s call the `out` *Tensor*  $ o$. We have that $ o = \frac{1}{4}\sum_i z_i$,$z_i = 3(x_i+2)^2$ and $ z_i\bigr\rvert_{x_i=1} = 27$. Therefore, $ \frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)$, hence $ \frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5$.
      + 向量的情况
         ```python
         x = torch.randn(3, requires_grad=True)

         y = x * 2
         while y.data.norm() < 1000:   # 计算y的2范数
            y = y * 2

         print(y)
         # tensor([ 100.8142, 1115.5734,    7.2186], grad_fn=<MulBackward0>)
         
         v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
         y.backward(v)

         print(x.grad)
         tensor([5.1200e+01, 5.1200e+02, 5.1200e-02])
         ```
      + Mathematically, if you have a vector valued function $\vec{y}=f(\vec{x})$,then the gradient of $\vec{y}$ with respect to $\vec{x}$ is a Jacobian matrix:

         $$
         J
            =
               \left(\begin{array}{cc}
               \frac{\partial \bf{y}}{\partial x_{1}} &
               ... &
               \frac{\partial \bf{y}}{\partial x_{n}}
               \end{array}\right)
            =
            \left(\begin{array}{ccc}
               \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
               \vdots & \ddots & \vdots\\
               \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
               \end{array}\right)
         $$

         ​Generally speaking, ``torch.autograd`` is an engine for computing vector-Jacobian product. That is, given any vector $v=\left(\begin{array}{cccc} v_{1} & v_{2} & \cdots & v_{m}\end{array}\right)^{T}$, compute the product $v^{T}\cdot J$. If $v$ happens to be the gradient of a scalar function $l=g\left(\vec{y}\right)$, that is, $v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}$, then by the chain rule, the vector-Jacobian product would be the
         gradient of $l$ with respect to $\vec{x}$:
         $$
         J^{T}\cdot v=\left(\begin{array}{ccc}
            \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
            \vdots & \ddots & \vdots\\
            \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
            \end{array}\right)\left(\begin{array}{c}
            \frac{\partial l}{\partial y_{1}}\\
            \vdots\\
            \frac{\partial l}{\partial y_{m}}
            \end{array}\right)=\left(\begin{array}{c}
            \frac{\partial l}{\partial x_{1}}\\
            \vdots\\
            \frac{\partial l}{\partial x_{n}}
            \end{array}\right)
         $$
         (Note that $v^{T}\cdot J$ gives a row vector which can be treated as a column vector by taking $J^{T}\cdot v$.) This characteristic of vector-Jacobian product makes it very convenient to feed external gradients into a model that has non-scalar output.
      

            ```python
            print(x.requires_grad)
            # True
            print((x ** 2).requires_grad)
            # True

            with torch.no_grad():
               print((x ** 2).requires_grad)
            # False
            
            y = x.detach()
            print(y.requires_grad)
            # False
            print(x.eq(y).all())
            # tensor(True)
            ```
      + 计算图： DAGs are dynamic in PyTorch An important thing to note is that the graph is recreated from scratch; after each .backward() call, autograd starts populating a new graph. This is exactly what allows you to use control flow statements in your model; you can change the shape, size and operations at every iteration if needed.

      + [More autograd.Function](https://pytorch.org/docs/stable/autograd.html#function)
----
##### Neural Networks
----
   + 神经网络可以使用`torch.nn`包来构建
   + A typical training procedure for a neural network is as follows:

      1. Define the neural network that has some learnable parameters (or weights)
      2. Iterate over a dataset of inputs
      3. Process input through the network
      4. Compute the loss (how far is the output from being correct)
      4. Propagate gradients back into the network’s parameters
      5. Update the weights of the network, typically using a simple update rule: **weight = weight - learning_rate * gradient**
   1. 网络结构定义
   ```python
   # 一个简单CNN的定义 
   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           # 1 input image channel, 6 output channels, 3x3 square convolution
           # kernel
           self.conv1 = nn.Conv2d(1, 6, 3)
           self.conv2 = nn.Conv2d(6, 16, 3)
           # an affine operation: y = Wx + b
           self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension 
           self.fc2 = nn.Linear(120, 84)
           self.fc3 = nn.Linear(84, 10)

       def forward(self, x):
           # Max pooling over a (2, 2) window
           x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
           # If the size is a square you can only specify a single number
           x = F.max_pool2d(F.relu(self.conv2(x)), 2)
           x = x.view(-1, self.num_flat_features(x))
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           x = self.fc3(x)
           return x

       def num_flat_features(self, x):
           size = x.size()[1:]  # all dimensions except the batch dimension
           num_features = 1
           for s in size:
               num_features *= s
           return num_features

   net = Net()
   print(net)
   
   # Net(
  # (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  # (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  # (fc1): Linear(in_features=576, out_features=120, bias=True)
  # (fc2): Linear(in_features=120, out_features=84, bias=True)
  # (fc3): Linear(in_features=84, out_features=10, bias=True)
   ```
   + 只需定义`forward()`，然后使用autograd自动定义backward()(其中计算梯度)。可以在forward()中使用任何一个张量运算。
   + 模型的可学习参数由`net.parameters()`返回。
   ```python
   params = list(net.parameters())
   print(len(params))
   # 10
   print(params[0].size())   # conv1's .weight
   # torch.Size([6, 1, 3, 3])  
   ```
   2. 前向计算
   ```python
   input = torch.rand(1, 1, 32, 32)
   out = net(input)
   print(out)
   # tensor([[-0.0067, -0.0947,  0.0745,  0.1046, -0.0269,  0.0712,  0.1327,  0.0712,
   #          -0.0826,  0.0740]], grad_fn=<AddmmBackward>)
   ```
   3. 后向计算 所有参数的梯度缓冲区归零
   ```python
   net.zero_grad()
   out.backward(torch.randn(1, 10))
   ```
   4. 整个`torch.nn`只支持小批量样本的输入，而不是单个样本; **如果是单个样本, 使用`input.unsqueeze(0)`添加一个假的miniBatch**
   5. `nn.Conv2d()`接受的是4维张量, `nSamples x nChannels x Height x Width`
   6. 损失函数
      + 在nn包下有几个不同的损失函数。一个简单的损失是:`nn.MSELoss`计算输入和目标之间的均方误差。
      ```python
      output = net(input)
      target = torch.rand(10)
      target = target.view(1, -1)
      
      criterion = nn.MSELoss()
      loss = criterion(output, target)
      print(loss)
      # tensor(1.1995, grad_fn=<MseLossBackward>)
      ```
      + 现在，如果使用`.grad_fn`属性，按照`backward()`方向跟踪loss，您将看到一个计算图，如下所示:
      ```
      input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
      ```
      ```python
      print(loss.grad_fn)  # MSELoss
      # <MseLossBackward object at 0x7f6d9e7788d0>
      print(loss.grad_fn.next_functions[0][0])  # Linear
      # <AddmmBackward object at 0x7f6d9e7784a8>
      print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
      ```
   7. 后向计算
      + 您需要清除现有的梯度，否则梯度将积累到现有的梯度
      ```python
      net.zero_grad()
      ```
      + 后向计算
      ```python
      print('conv1.bias.grad before backward')
      print(net.conv1.bias.grad)
      loss.backward()
      print('conv1.bias.grad after backward')
      print(net.conv1.bias.grad)
      # conv1.bias.grad before backward
      # tensor([0., 0., 0., 0., 0., 0.])
      # conv1.bias.grad after backward
      # tensor([-0.0022, -0.0174, -0.0071, -0.0145,  0.0047,  0.0032])
      ```
   + [More lossfunction](https://pytorch.org/docs/nn)
   8. 更新参数
      + 随机梯度下降
      ```python
      learning_rate = 0.01
      for f in net.parameters():
          f.data.sub_(f.grad.data*learning_rate)
      ```
      + 使用`torch.optim`包
      ```python
      import torch.optim as optim

      # create your optimizer
      optimizer = optim.SGD(net.parameters(). lr=0.01)
      # in your training loop:
      optimizer.zero_grad()  # zero the gradient buffers
      output = net(input)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()  # Does the update     
      ```
      + 使用optimizer.zero_grad()手动将梯度缓冲区设置为零, 这是因为梯度是累积的
----
##### Training a Classifier
----
   1. 加载和正则化数据
   ```python
   import torch
   import torchvision
   import torchvision.transforms as transforms
   
   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

   trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

   testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
   testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

   classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
   ```
   2. 可视化图片
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   
   def imgshow(img):
      img = img / 2 + 0.5
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))
      plt.show()

   # get some random training images
    = iter(trainloader)
   images, labels = dataiter.next()

   # show images
   imshow(torchvision.utils.make_grid(images))
   # print labels
   print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
   ```
   3. 定义卷积网络, 此时为3通道网络
   ```python
   import torch
   import torch.nn as nn 
   import torch.nn.functional as f
   
   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.conv1 = nn.Conv2d(3, 6, 5)
           self.pool = nn.MaxPool2d(2, 2)
           self.conv2 = nn.conv2d(6, 16, 5)
           self.fc1 = nn.Linear(16 * 5 * 5, 120)
           self.fc2 = nn.Linear(120, 84)
           self.fc3 = nn.Linear(84, 10)
    
       def forward(self, x):
           x = self.pool(f.relu(self.conv1(x)))
           x = self.pool(f.relu(self.conv2(x)))
           x = x.view(-1, 16 * 5 * 5)
           x = f.relu(self.fc1(x))
           x = f.relu(self.fc2(x))
           x = self.fc3(x)
           return x

   net = Net()
   ```
   4. 定义损失函数和优化器
      + 交叉熵损失函数, SGD优化器, 动量项
   ```python
   import torch.optim as optim
    
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
   ```
   5. 训练网络
   ```python
   for epoch in range(2):  # loop over the dataset multiple times
       running_loss = 0
       for i, data in enumerate(trainloader, 0):
           # get the inputs; data is a list of [inputs, labels]
           inputs, labels = data
           
           # zero the parameter gradients
           optimizer.zero_grad()
           
           # forward + backward + optimize
           output = net(inputs)
           loss = criterion(output, labels)
           loss.backward()
           optimizer.step()  # update
           
           # print statistics
           running_loss += loss.item()
           if i%2000 == 1999:   # print every 2000 mini-batches
               print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
               running_loss = 0.0  
   print("Training Finished")
   ```
   6. 保存网络结构
   ```python
   PATH = "./cifar_net.pth"
   torch.save(net.state_dict(), PATH)
   ```
   + [for more details on saving PyTorch models.](https://pytorch.org/docs/stable/notes/serialization.html)
   7. 加载模型, 测试数据测试网络
   ```python
   # load test images
   dataiter = iter(testloader)
   images, labels = dataiter.next()
   
   # print images
   imshow(torchvision.utils.make_grid(images))
   print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
   
   # load module
   net = Net()
   net.load_state_dict(torch.load(PATH))
   
   output = net(images)
   
   _, predicted = torch.max(output, 1)
   print("predicted, ", " ".join('%5s' % classes[predicted[j]] for j in range(4)))
   ```

   ```python
   # 在整个test_data上测试
   correct = 0
   total = 0
   with torch.no_grad():
       for data in testloader:
           images, labels = data
           outputs = net(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()

   print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
   ```
   8. 使用GPU
   ```python
   device = torch.device("CUDA:0" if torch.cuda.is_available() else "cpu")
   # Assuming that we are on a CUDA machine, this should print a CUDA device:
   print(device) 
   ```
   + ***请记住，必须将input和target在每一步也输入GPU***
   ```python
   net.to(device)
   inputs, labels = data[0].to(device), data[1].to(device)
   ```
   
      
      



### Learning Pytorch with Examples
---


### What is *torch.nn* really?
---
 

### Visualizing Models, Data, and Training with TensorBoard
---

## Chapter 2 Image/Video
---

---

----
### Chapter 2 Learning PyTorch With Examples
----
   1. Tensor
      + PyTorch张量与numpy数组在概念上是相同的:张量是一个n维数组，PyTorch提供了许多操作这些张量的函数。在幕后，张量可以跟踪计算图形和梯度，但是它们作为科学计算的通用工具也很有用。与numpy不同的是，PyTorch张量可以利用gpu加速数值计算。要在GPU上运行一个PyTorch张量，你只需要将它转换成一个新的数据类型。 
      + numpy实现两层网络结构 
      ```python
      # -*- coding: utf-8 -*-
      import numpy as np

      # N is batch size; D_in is input dimension;
      # H is hidden dimension; D_out is output dimension.
      N, D_in, H, D_out = 64, 1000, 100, 10

      # Create random input and output data
      x = np.random.randn(N, D_in)
      y = np.random.randn(N, D_out)

      # Randomly initialize weights
      w1 = np.random.randn(D_in, H)
      w2 = np.random.randn(H, D_out)

      learning_rate = 1e-6
      for t in range(500):
          # Forward pass: compute predicted y
          h = x.dot(w1)
          h_relu = np.maximum(h, 0)
          y_pred = h_relu.dot(w2)

          # Compute and print loss
          loss = np.square(y_pred - y).sum()
          print(t, loss)

          # Backprop to compute gradients of w1 and w2 with respect to loss
          grad_y_pred = 2.0 * (y_pred - y)
          grad_w2 = h_relu.T.dot(grad_y_pred)
          grad_h_relu = grad_y_pred.dot(w2.T)
          grad_h = grad_h_relu.copy()
          grad_h[h < 0] = 0
          grad_w1 = x.T.dot(grad_h)

          # Update weights
          w1 -= learning_rate * grad_w1
          w2 -= learning_rate * grad_w2
      ```
      + PyTorch实现两层网络结构
      ```python
      import torch

      datatype = torch.float
      device = torch.device("cpu")
      # device = torch.device("cuda:0") # Uncomment this to run on GPU

      # N is batch size; D_in is input dimension;
      # H is hidden dimension; D_out is output dimension.
      N, D_in, H, D_out = 64, 1000, 100, 10
      # Create random input and output data
      x = torch.randn(N, D_in, device=device, dtype=datatype)
      y = torch.randn(N, D_out, device=device, dtype=datatype)

      # Randomly initialize weights
      w1 = torch.randn(D_in, H, device=device, dtype=datatype)
      w2 = torch.randn(H, D_out, device=device, dtype=datatype)

      learnint_rate = 1e-6
      for t in range(500):
          # Forward pass: compute predicted y
          h = x.mm(w1)
          h_relu = h.clamp(min=0)
          y_pred = h.relu.mm(w2)
          
          # Forward pass: compute predicted y
          loss = (y - y_pred).pow(2).sum().item()
          print(t, loss)
          
          # Backprop to compute gradients of w1 and w2 with respect to loss
          grad_y_pred = 2 * (y - y_pred) 
          grad_w2 = h_relu.t().mm(grad_y_pred)
          grad_h_relu = grad_y_pred.mm(w2.t())
          grad_h = grad_h_relu.clone()
          grad_h[h < 0] = 0
          grad_w1 = x.t().mm(grad_h)

          # Update weights using gradient descent
          w1 -= learning_rate * grad_w1
          w2 -= learning_rate * grad_w2
      ```
   2. Autograd
      + 使用autograd时，网络的前向传递将定义一个计算图;图中的节点是张量，边是由输入张量产生输出张量的函数。通过这个图的反向传播可以轻松地计算梯度。
      + 每个张量表示计算图中的一个节点。如果x是一个`x.requires_grad=True`的张量。然后`x.grad`是另一个张量, 包含x的梯度。
      ```python
      # -*- coding: utf-8 -*-
      import torch

      dtype = torch.float
      device = torch.device("cpu")
      # device = torch.device("cuda:0") # Uncomment this to run on GPU

      # N is batch size; D_in is input dimension;
      # H is hidden dimension; D_out is output dimension.
      N, D_in, H, D_out = 64, 1000, 100, 10

      # Create random Tensors to hold input and outputs.
      # Setting requires_grad=False indicates that we do not need to compute gradients
      # with respect to these Tensors during the backward pass.
      x = torch.randn(N, D_in, device=device, dtype=dtype)
      y = torch.randn(N, D_out, device=device, dtype=dtype)

      # Create random Tensors for weights.
      # Setting requires_grad=True indicates that we want to compute gradients with
      # respect to these Tensors during the backward pass.
      w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
      w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

      learning_rate = 1e-6
      for t in range(500):
          # Forward pass: compute predicted y using operations on Tensors; these
          # are exactly the same operations we used to compute the forward pass using
          # Tensors, but we do not need to keep references to intermediate values since
          # we are not implementing the backward pass by hand.
          y_pred = x.mm(w1).clamp(min=0).mm(w2)

          # Compute and print loss using operations on Tensors.
          # Now loss is a Tensor of shape (1,)
          # loss.item() gets the scalar value held in the loss.
          loss = (y_pred - y).pow(2).sum()
          if t % 100 == 99:
              print(t, loss.item())

          # Use autograd to compute the backward pass. This call will compute the
          # gradient of loss with respect to all Tensors with requires_grad=True.
          # After this call w1.grad and w2.grad will be Tensors holding the gradient
          # of the loss with respect to w1 and w2 respectively.
          loss.backward()

          # Manually update weights using gradient descent. Wrap in torch.no_grad()
          # because weights have requires_grad=True, but we don't need to track this
          # in autograd.
          # An alternative way is to operate on weight.data and weight.grad.data.
          # Recall that tensor.data gives a tensor that shares the storage with
          # tensor, but doesn't track history.
          # You can also use torch.optim.SGD to achieve this.
          with torch.no_grad():
              w1 -= learning_rate * w1.grad
              w2 -= learning_rate * w2.grad

              # Manually zero the gradients after updating weights
              w1.grad.zero_()
              w2.grad.zero_()
      ```
      + 自定义函数, 实现自动微分; 创建子类`torch.autograd.Function`, 并在类中实现`forward(), backward()`方法, 使用时`FunctionName.apply()`
      ```python
      # -*- coding: utf-8 -*-
      import torch

      class MyRelu(torch.autograd.Function):
         """
         We can implement our own custom autograd Functions by subclassing
         torch.autograd.Function and implementing the forward and backward passes
         which operate on Tensors.
         """
         @staticmethod
         def forward(ctx, input):
            """
            In the forward pass we receive a Tensor containing the input and return
            a Tensor containing the output. ctx is a context object that can be used
            to stash information for backward computation. You can cache arbitrary
            objects for use in the backward pass using the ctx.save_for_backward method.
            """
            ctx.save_for_backward(input)
            return input.clamp(min=0)

         def backward(ctx, grad_ouput):
            """
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
            """
            input, =ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input<0] = 0
            return grad_input

      dtype = torch.float
      device = torch.device("cpu")
      # device = torch.device("cuda:0") # Uncomment this to run on GPU

      # N is batch size; D_in is input dimension;
      # H is hidden dimension; D_out is output dimension.
      N, D_in, H, D_out = 64, 1000, 100, 10

      # Create random Tensors to hold input and outputs.
      x = torch.randn(N, D_in, device=device, dtype=dtype)
      y = torch.randn(N, D_out, device=device, dtype=dtype)

      # Create random Tensors for weights.
      w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
      w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

      learning_rate = 1e-6
      for t in range(500):
          # To apply our Function, we use Function.apply method. We alias this as 'relu'.
          relu = MyReLU.apply

          # Forward pass: compute predicted y using operations; we compute
          # ReLU using our custom autograd operation.
          y_pred = relu(x.mm(w1)).mm(w2)

          # Compute and print loss
          loss = (y_pred - y).pow(2).sum()
          if t % 100 == 99:
              print(t, loss.item())

          # Use autograd to compute the backward pass.
          loss.backward()
      
          # Update weights using gradient descent
          with torch.no_grad():
              w1 -= learning_rate * w1.grad
              w2 -= learning_rate * w2.grad

              # Manually zero the gradients after updating weights
              w1.grad.zero_()
              w2.grad.zero_()
      ```
   3. nn module
      + nn package定义了一组模块，大致相当于神经网络层。模块接收输入张量并计算输出张量，但也可以保存内部状态，如包含可学习参数的张量。神经网络包还定义了一组有用的损失函数，这些函数通常用于训练神经网络。
      ```python
      # -*- coding: utf-8 -*-
      import torch

      dtype = torch.float
      device = torch.device('cpu')

      # N is batch size; D_in is input dimension;
      # H is hidden dimension; D_out is output dimension.
      N, D_in, H, D_out = 64, 1000, 100, 10

      # Create random Tensors to hold inputs and outputs
      x = torch.randn(N, D_in, device=device, dtype=dtype)
      y = torch.randn(N, D_out, device=device, dtype=dtype)

      # Use the nn package to define our model as a sequence of layers. nn.Sequential
      # is a Module which contains other Modules, and applies them in sequence to
      # produce its output. Each Linear Module computes output from input using a
      # linear function, and holds internal Tensors for its weight and bias.
      model = torch.nn.Sequential(
         torch.nn.Linear(D_in, H)
         torch.nn.Relu()
         torch.nn.Linear(H, D_out)
      )
      # The nn package also contains definitions of popular loss functions; in this
      # case we will use Mean Squared Error (MSE) as our loss function.
      loss_fn = torch.nn.MSELoss(reduction='sum')

      learning_rate = 1e-4
      for t in range(500):
          # Forward pass: compute predicted y by passing x to the model. Module objects
          # override the __call__ operator so you can call them like functions. When
          # doing so you pass a Tensor of input data to the Module and it produces
          # a Tensor of output data.
          y_pred = model(x)

          # Compute and print loss. We pass Tensors containing the predicted and true
          # values of y, and the loss function returns a Tensor containing the
          # loss.
          loss = loss_fn(y_pred, y)
          if t % 100 == 99:
              print(t, loss.item())

          # Zero the gradients before running the backward pass.
          model.zero_grad()

          # Backward pass: compute gradient of the loss with respect to all the learnable
          # parameters of the model. Internally, the parameters of each Module are stored
          # in Tensors with requires_grad=True, so this call will compute gradients for
          # all learnable parameters in the model.
          loss.backward()

          # Update the weights using gradient descent. Each parameter is a Tensor, so
          # we can access its gradients like we did before.
          with torch.no_grad():
              for param in model.parameters():
                  param -= learning_rate * param.grad
      ```
      + PyTorch中的optim包抽象了优化算法的思想，并提供了常用优化算法的实现。(使用`torch.no_grad()`或`.data`来避免跟踪autograd中的历史记录)
      ```python
      # -*- coding: utf-8 -*-
      import torch

      # N is batch size; D_in is input dimension;
      # H is hidden dimension; D_out is output dimension.
      N, D_in, H, D_out = 64, 1000, 100, 10

      # Create random Tensors to hold inputs and outputs
      x = torch.randn(N, D_in)
      y = torch.randn(N, D_out)
      
      # Use the nn package to define our model and loss function.
      model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
      )
      loss_fn = torch.nn.MSELoss(reduction='sum')

      # Use the optim package to define an Optimizer that will update the weights of
      # the model for us. Here we will use Adam; the optim package contains many other
      # optimization algoriths. The first argument to the Adam constructor tells the
      # optimizer which Tensors it should update.
      learning_rate = 1e-4
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      for t in range(500):
          # Forward pass: compute predicted y by passing x to the model.
          y_pred = model(x)

          # Compute and print loss.
          loss = loss_fn(y_pred, y)
          if t % 100 == 99:
              print(t, loss.item())

          # Before the backward pass, use the optimizer object to zero all of the
          # gradients for the variables it will update (which are the learnable
          # weights of the model). This is because by default, gradients are
          # accumulated in buffers( i.e, not overwritten) whenever .backward()
          # is called. Checkout docs of torch.autograd.backward for more details.
          optimizer.zero_grad()

          # Backward pass: compute gradient of the loss with respect to model
          # parameters
          loss.backward()

          # Calling the step function on an Optimizer makes an update to its
          # parameters
          optimizer.step()
      ```
      + 自定义训练模型 Custiom nn Modules, 有时可能希望指定比现有模块序列更复杂的模型; 对于这些情况，您可以通过生成`torch.nn.Module`子类来定义自己的模块。定义`forward()`一个接收输入张量并使用其他模块或其他自动获取张量的操作生成输出张量
      ```python 
      # -*- coding: utf-8 -*-
      import torch

      class TwoLayerNet(torch.nn.Module):
         def __init__(self, D_in, H, D_out):
            """
            In the constructor we instantiate two nn.Linear modules and assign them as
            member variables.
            """
            super(TwoLayerNet, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, D_out)
         
         def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred

      # N is batch size; D_in is input dimension;
      # H is hidden dimension; D_out is output dimension.
      N, D_in, H, D_out = 64, 1000, 100, 10

      # Create random Tensors to hold inputs and outputs
      x = torch.randn(N, D_in)
      y = torch.randn(N, D_out)

      # Construct our model by instantiating the class defined above
      model = TwoLayerNet(D_in, H, D_out)

      # Construct our loss function and an Optimizer. The call to model.parameters()
      # in the SGD constructor will contain the learnable parameters of the two
      # nn.Linear modules which are members of the model.
      criterion = torch.nn.MSELoss(reduction='sum')
      optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
      for t in range(500):
          # Forward pass: Compute predicted y by passing x to the model
          y_pred = model(x)

          # Compute and print loss
          loss = criterion(y_pred, y)
          if t % 100 == 99:
              print(t, loss.item())

          # Zero gradients, perform a backward pass, and update the weights.
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()  
      ```
      + 权重共享, 模型多次复用
      ```python
      # -*- coding: utf-8 -*-
      import random
      import torch

      class DynamicNet(torch.nn.Module):
          def __init__(self, D_in, H, D_out):
              """
              In the constructor we construct three nn.Linear instances that we will use
              in the forward pass.
              """
              super(DynamicNet, self).__init__()
              self.input_linear = torch.nn.Linear(D_in, H)
              self.middle_linear = torch.nn.Linear(H, H)
              self.output_linear = torch.nn.Linear(H, D_out)

          def forward(self, x):
              """
              For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
              and reuse the middle_linear Module that many times to compute hidden layer
              representations.

              Since each forward pass builds a dynamic computation graph, we can use normal
              Python control-flow operators like loops or conditional statements when
              defining the forward pass of the model.

              Here we also see that it is perfectly safe to reuse the same Module many
              times when defining a computational graph. This is a big improvement from Lua
              Torch, where each Module could be used only once.
              """
               h_relu = self.input_linear(x).clamp(min=0)
              for _ in range(random.randint(0, 3)):
                  h_relu = self.middle_linear(h_relu).clamp(min=0)
              y_pred = self.output_linear(h_relu)
              return y_pred
      ```

      ```python
      # N is batch size; D_in is input dimension;
      # H is hidden dimension; D_out is output dimension.
      N, D_in, H, D_out = 64, 1000, 100, 10
        
      # Create random Tensors to hold inputs and outputs
      x = torch.randn(N, D_in)
      y = torch.randn(N, D_out)
        
      # Construct our model by instantiating the class defined above
      model = DynamicNet(D_in, H, D_out)
        
      # Construct our loss function and an Optimizer. Training this strange model with
      # vanilla stochastic gradient descent is tough, so we use momentum
      criterion = torch.nn.MSELoss(reduction='sum')
      optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
      for t in range(500):
          # Forward pass: Compute predicted y by passing x to the model
          y_pred = model(x)
        
          # Compute and print loss
          loss = criterion(y_pred, y)
          if t % 100 == 99:
              print(t, loss.item())
        
          # Zero gradients, perform a backward pass, and update the weights.
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      ```
----
### Chapter 3 What's torch.nn really?
----
   1. Neural Net from scratch
      + 权重和偏置初始化 
      ```python
      import math

      weights = torch.randn(784, 10) / math.sqrt(784)
      weights.requires_grad_()
      bias = torch.zeros(10, requires_grad=True)
      ```
      + 定义模型   `@`代表点乘
      ```python
      def log_softmax(x)
         return x - x.exp().sum(-1).log().unsqueeze(-1)
      
      def model(xb):
         return log_softmax(xb @ weights + bias)
      ```
      + 前向计算
      ```python
      bs = 64  # batch size

      xb = x_train[0:bs]  # a mini-batch from x
      preds = model(xb)  # predictions
      preds[0], preds.shape
      print(preds[0], preds.shape)
      ```
      + loss function
      ```python
      def nll(input, target):
         return -input[range(target.shape[0]), target].mean()

      loss_func = nll
      yb = y_train[0:bs]
      print(loss_func(preds, yb))

      def accuracy(out, yb):
          preds = torch.argmax(out, dim=1)
          return (preds == yb).float().mean()
      print(accuracy(preds, yb))
      ```
      + training loop 训练需要在`torch.no_grad()`中更新参数, 因为这些操作不需要记录到梯度计算中, 下一轮的训练开始前需要清空梯度
      ```python
      from IPython.core.debugger import set_trace

      lr = 0.5  # learning rate
      epochs = 2  # how many epochs to train for

      for epoch in range(epochs):
          for i in range((n - 1) // bs + 1):
             #         set_trace()
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                  weights -= weights.grad * lr
                  bias -= bias.grad * lr
                  weights.grad.zero_()
                  bias.grad.zero_()
      ```
   2. 使用torch.nn.functional
      + 这个模块包含了`torch.nn`中的所有函数, (而库的其他部分包含类)。除了各种各样的损失和激活函数之外，您还可以在这里找到一些用于创建神经网络的方便函数，比如池函数, 也有一些函数用于执行卷积、线性层等 
      ```python
      import torch.nn.functional as F

      loss_func = F.cross_entropy
      def model(xb):
         return xb @ weights + bias
      ```
   3. 使用nn.Module和nn.Parameter
      ```python
      from torch import nn

      class Mnist_Logistic(nn.Module):
          def __init__(self):
              super().__init__()
              self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
              self.bias = nn.Parameter(torch.zeros(10))

          def forward(self, xb):
             return xb @ self.weights + self.bias
      
      model = Mnist_Logistic()
      def fit():
          for epoch in range(epochs):
              for i in range((n - 1) // bs + 1):
                  start_i = i * bs
                  end_i = start_i + bs
                  xb = x_train[start_i:end_i]
                  yb = y_train[start_i:end_i]
                  pred = model(xb)
                  loss = loss_func(pred, yb)

                  loss.backward()
                  with torch.no_grad():
                     for p in model.parameters():
                        p -= p.grad * lr
                     model.zero_grad()
      ```
   4. 使用nn.Linear()
   ```python
   class Mnist_Logistic(nn.Module):
       def __init__(self):
           super().__init__()
           self.lin = nn.Linear(784, 10)

       def forward(self, xb):
           return self.lin(xb)
   ```
   5. 使用 optim
   ```python
   def get_model():
      model = Mnist_Logistic()
      return model, optim.SGD(model.parameters(), lr=lr)

   model, opt = get_model()
   print(loss_func(model(xb), yb))

   for epoch in range(epochs):
      for i in range((n - 1) // bs + 1):
         start_i = i * bs
         end_i = start_i + bs
         xb = x_train[start_i:end_i]
         yb = y_train[start_i:end_i]
         pred = model(xb)
         loss = loss_func(pred, yb)

         loss.backward()
         opt.step()
         opt.zero_grad()

   print(loss_func(model(xb), yb))
   ```
   6. 使用Dataset
      + PyTorch有一个抽象的Dataset类。一个Dataset可以是任何具有一个`__len__`函数(由Python的标准len函数调用)和一个`__getitem__`函数作为索引的数据集; PyTorch的TensorDataset是一个包装张量的数据集; 通过定义长度和索引的方式，这也给了我们一种方法来遍历、索引和切片一个张量的维度
      + Both `x_train` and `y_train` can be combined in a single TensorDataset, which will be easier to iterate over and slice.
      ```python
      from torch.utils.data import TensorDataset

      train_ds = TensorDataset(x_train, y_train)

      xb,yb = train_ds[i*bs : i*bs+bs]
      model, opt = get_model()

      for epoch in range(epochs):
         for i in range((n - 1) // bs + 1):
            xb, yb = train_ds[i * bs: i * bs + bs]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

      print(loss_func(model(xb), yb))
      ```
   7. 使用DataLoader
      + Pytorch的DataLoader负责管理batches, 可以从任何Dataset创建DataLoader, DataLoader简化了对batch的迭代, 而不是必须使用train_ds[i*bs: i*bs+bs], DataLoader自动为我们提供每个minibatch
      ```python
      from torch.utils.data import DataLoader

      train_ds = TensorDataset(x_train, y_train)
      train_dl = DataLoader(train_ds, batch_size=bs)
      # for xb,yb in train_dl:
      #    pred = model(xb)

      for epoch in range(epochs):
         for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

      print(loss_func(model(xb), yb))
      ```
   8. 添加Validation
   ```python
   train_ds = TensorDataset(x_train, y_train)
   train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

   valid_ds = TensorDataset(x_valid, y_valid)
   valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
   ```
   + 注意，我们总是在训练之前调用`model.train()`，在inference之前调用`model.eval()`，因为这些都是由类似`nn.BatchNorm2d, nn.Dropout`使用的, 以确保在训练和测试阶段的适当的行为
      ```python
      model, opt = get_model()
        
      for epoch in range(epochs):
         model.train()
         for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
        
            loss.backward()
            opt.step()
            opt.zero_grad()
        
         model.eval()
         with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
        
         print(epoch, valid_loss / len(valid_dl))
      ```
   9. 整合代码, 创建fit(), get_data()
      + 创建loss_batch, 可以简化训练和验证的误差计算
      ```python
      def loss_batch(model, xb, yb, loss_func, opt=None):
         loss = loss_func(model(xb), yb)
         if opt is Not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
         return loss.item(), len(xb)
      
      import numpy as np

      def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
         for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                  loss_batch(model, loss_func, xb, yb, opt)

            model.eval()
            with torch.no_grad():
                  losses, nums = zip(
                     *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
                  )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            print(epoch, val_loss)
      ```
      + get_data(), returns dataloaders for the training and validation sets.
      ```python
      def get_data(train_ds, valid_ds, bs):
         return (
            DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(valid_ds, batch_size=bs * 2),
         )
      ```
      + 所有代码
      ```python
      train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
      model, opt = get_model()
      fit(epochs, model, loss_func, opt, train_dl, valid_dl)
      ```
   10. 建立CNN
   ```python
   class Mnist_CNN(nn.Module):
      def __init__(self):
         super().__init__()
         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
         self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
         self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

      def forward(self, xb):
         xb = xb.view(-1, 1, 28, 28)
         xb = F.relu(self.conv1(xb))
         xb = F.relu(self.conv2(xb))
         xb = F.relu(self.conv3(xb))
         xb = F.avg_pool2d(xb, 4)
         return xb.view(-1, xb.size(1))

   lr = 0.1
   ```
   11. nn.Sequential()

   + Sequential对象以顺序的方式运行其中包含的每个模块, 这是一种更简单的神经网络的写法
   + 为了利用Sequential，我们需要能够从一个给定的函数轻松地定义一个自定义层, 例如，PyTorch没有view层，我们需要为我们的网络创建一个view层。Lambda将创建一个层，然后我们可以使用它来定义一个Sequential网络
      ```python
      def preprocess(x):
         return x.view(-1, 1, 28, 28)
      
      class Lambda(nn.Module):
         def __init__(self, func):
            self.func = func
         
         def forward(self, x):
            return self.func(x)
      
      model = nn.Sequential(
         Lambda(preprocess),
         nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
         # 需要将relu函数改为layer
         nn.ReLU(),
         nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
         nn.ReLU(),
         nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
         nn.ReLU(),
         nn.AvgPool2d(4),
         Lambda(lambda x: x.view(x.size(0), -1)),
      )
        
      opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
      fit(epochs, model, loss_func, opt, train_dl, valid_dl)
      ```
   12. Wrapping DataLoader
   ```python
   def preprocess(x, y):
      return x.view(-1, 1, 28, 28), y

   class WrappedDataLoader:
      def __init__(self, dl, func):
         self.dl = dl
         self.func = func

      def __len__(self):
         return len(self.dl)

      def __iter__(self):
         batches = iter(self.dl)
         for b in batches:
               yield (self.func(*b))

   train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
   train_dl = WrappedDataLoader(train_dl, preprocess)
   valid_dl = WrappedDataLoader(valid_dl, preprocess)
   ```
   + 可以替换nn.AvgPool2d为nn.adaptiveeavgpool2d, 它允许我们定义我们想要的输出张量的大小, `nn.adaptiveeavgpool2d((H,W))` 
      ```python
      model = nn.Sequential(
         nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
         nn.ReLU(),
         nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
         nn.ReLU(),
         nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
         nn.ReLU(),
         nn.AdaptiveAvgPool2d(1),
         Lambda(lambda x: x.view(x.size(0), -1)),
      )
        
      opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
      ```
   13. 使用GPU
   ```python
   dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

   def preprocess(x, y):
      # 把张量转到GPU上
      return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


   train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
   train_dl = WrappedDataLoader(train_dl, preprocess)
   valid_dl = WrappedDataLoader(valid_dl, preprocess)

   # 模型也需要在GPU上
   model.to(dev)
   opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
   ```

----
### Chapter 4 Text
----
#### Sequence-to-Sequence Modeling with nn.Transformer and TorchText
   + `nn.Transformer, nn.MultiheadAttention, nn.TransformerEncoder`
      ![transformer_architecture](C:\Users\lenovo\Desktop\learn_pytorch\NLPwithPyTorch\transformer_architecture.jpg)
   1. 定义模型 构建一个Transformer模型来构建语言模型任务
      + `nn.TransformerEncoder`由多层`nn.TransformerEncoderLayer`组成。ransformerEncoder只允许出现在序列中较早的位置。
      1. `torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None)`
         + d_model: the number of expected features in the encoder/decoder inputs (default=512).
         + nhead: the number of heads in the multiheadattention models (default=8).
         + num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
         + num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
         + dim_feedforward: the dimension of the feedforward network model (default=2048).
         + dropout: the dropout value (default=0.1).
         + activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
         + custom_encoder: custom encoder (default=None).
         + custom_decoder: custom decoder (default=None).
         
      2. `torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)`, N个编码层的堆栈
         + encoder_layer: an instance of the TransformerEncoderLayer() class (required).
         + num_layers: the number of sub-encoder-layers in the encoder (required).
         + norm: the layer normalization component (optional).
         + `forward(src, mask=None, src_key_padding_mask=None)` 传递输入依次通过编码器层
            + src: the sequnce to the encoder (required).
            + mask: the mask for the src sequence (optional).
            + src_key_padding_mask: the mask for the src keys per batch (optional).
         
      3. `torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)` N个解码层的堆栈
         + `forward(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)` 传递输入依次通过解码层
            + tgt: the sequence to the decoder (required).
            + memory: the sequnce from the last layer of the encoder (required).
            + tgt_mask: the mask for the tgt sequence (optional)
            + memory_mask: the mask for the memory sequence (optional).
            + tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            + memory_key_padding_mask: the mask for the memory keys per batch (optional).
         
      4. `torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')`
      
      5. `torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')`
      
   2. torchtext, 获取数据
   ```python
   import torchtext
   from torchtext.data.utils import get_tokenizer

   TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
   train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
   TEXT.build_vocab(train_txt)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   def batchify(data, bsz):
      data = TEXT.numericalize([data.examples[0].text])
      # Divide the dataset into bsz parts.
      nbatch = data.size(0) // bsz
      # Trim off any extra elements that wouldn't cleanly fit (remainders).
      data = data.narrow(0, 0, nbatch * bsz)
      # Evenly divide the data across the bsz batches.
      data = data.view(bsz, -1).t().contiguous()
      return data.to(device)

   batch_size = 20
   eval_batch_size = 10
   train_data = batchify(train_txt, batch_size)
   val_data = batchify(val_txt, eval_batch_size)
   test_data = batchify(test_txt, eval_batch_size)
   ```

----
#### Classifying Names with A Character-Level RNN
----
   1. preparing the data, 读取文件, 从Unicode转化为Ascii
   ```python
   from __future__ import unicode_literals, print_function, division
   from io import open
   import glob
   import os

   def findFiles(path): return glob.glob(path)

   print(findFiles('data/names/*.txt'))

   import unicodedata
   import string

   all_letters = string.ascii_letters + " .,;'"
   n_letters = len(all_letters)

   # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
   def unicodeToAscii(s):
      return ''.join(
         c for c in unicodedata.normalize('NFD', s)
         if unicodedata.category(c) != 'Mn'
         and c in all_letters
      )

   print(unicodeToAscii('Ślusàrski'))

   # Build the category_lines dictionary, a list of names per language
   category_lines = {}
   all_categories = []

   # Read a file and split into lines
   def readLines(filename):
      lines = open(filename, encoding='utf-8').read().strip().split('\n')
      return [unicodeToAscii(line) for line in lines]

   for filename in findFiles('data/names/*.txt'):
      category = os.path.splitext(os.path.basename(filename))[0]
      all_categories.append(category)
      lines = readLines(filename)
      category_lines[category] = lines

   n_categories = len(all_categories)
   # category_lines, a dictionary mapping each category (language) to a list of lines (names).
   # all_categories (just a list of languages) and n_categories, the number of categories
   ```
   + Turning word to Tensor, 使用one-hot编码, 一个word就是一个二维矩阵, 再添加一个维度, batchsize
      ```python
      import torch
        
      # Find letter index from all_letters, e.g. "a" = 0
      def letterToIndex(letter):
         return all_letters.find(letter)
      
      # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
      def letterToTensor(letter):
         tensor = torch.zeros(1, n_letters)
         tensor[0][letterToIndex(letter)] = 1
         return tensor
      
      # Turn a line into a <line_length x 1 x n_letters>,
      # or an array of one-hot letter vectors
      def lineToTensor(line):
         tensor = torch.zeors(len(line), 1, n_letters)
         for index, letter in enumerate(line):
            tensor[index][0][letterToIndex(letter)] = 1
         return tensor
        
      print(letterToTensor('J'))
      print(lineToTensor('Jones').size())
      print(lineToTensor('Jones'))
      ```
   2. 创建网络结构, 只有一层RNN
   ```python
   import torch.nn as nn

   class RNN(nn.Module):
      def __init__(self, input_size, hidden_size, ouput_size):
         supper(RNN, self).__init__()
         self.hidden_size = hidden_size

         self.i2o = nn.Linear(input_size+hidden_size, output_size)
         self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
         self.softmax = nn.LogSoftmax(dim=1)
      
      def initHidden(self):
         return torch.zeros(1, self.hidden_size)
      
      def forward(self, input, hidden):
         combined = torch.cat((input, hidden), 1)
         output = self.i2o(combined)
         output = self.softmax(output)
         hidden = self.i2h(combined)
         return output, hidden

   n_hidden = 128
   rnn = RNN(n_letters, n_hidden, n_categories)
   input = lineToTensor('Albert')
   hidden = torch.zeros(1, n_hidden)

   output, next_hidden = rnn(input[0], hidden)
   print(output)
   ```
   3. 训练
      + 获取分类结果
      ```python
      def categoryFromOutput(output):
         top_value, top_index = output.topk(1)
         category_i = top_index[0].item()
         return all_categories[category_i], category_i

      print(categoryFromOutput(output))
      ```
      + 随机获取训练样本
      ```python
      import random

      def randomChoice(l):
         return l[random.randint(0, len(l) - 1)]

      def randomTrainingExample():
         category = randomChoice(all_categories)
         line = randomChoice(category_lines[category])
         category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
         line_tensor = lineToTensor(line)
         return category, line, category_tensor, line_tensor

      for i in range(10):
         category, line, category_tensor, line_tensor = randomTrainingExample()
         print('category =', category, '/ line =', line)
      ```
      + 训练过程
      ```python
      criterion = nn.NLLLoss()

      learning_rate = 0.005
      def train(category_tensor, line_tensor):
         hidden = rnn.initHidden()

         rnn.zero_grad()
         for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
         
         loss = criterion(output, category_tensor)
         loss.backward()
         
         for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)

         return output, loss.item()  

      import time
      import math

      n_iters = 100000
      print_every = 5000
      plot_every = 1000

      # Keep track of losses for plotting
      current_loss = 0
      all_losses = []
        
      def timeSince(since):
         now = time.time()
         s = now - since
         m = math.floor(s / 60)
         s -= m * 60
         return '%dm %ds' % (m, s)
        
      start = time.time()
        
      for iter in range(1, n_iters + 1):
         category, line, category_tensor, line_tensor = randomTrainingExample()
         output, loss = train(category_tensor, line_tensor)
         current_loss += loss
        
         # Print iter number, loss, name and guess
         if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        
         # Add current loss avg to list of losses
         if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0      
      ```
   4. 评估
   ```python
   # Keep track of correct guesses in a confusion matrix
   confusion = torch.zeros(n_categories, n_categories)
   n_confusion = 10000

   # Just return an output given a line
   def evaluate(line_tensor):
      hidden = rnn.initHidden()

      for i in range(line_tensor.size()[0]):
         output, hidden = rnn(line_tensor[i], hidden)

      return output

   # Go through a bunch of examples and record which are correctly guessed
   for i in range(n_confusion):
      category, line, category_tensor, line_tensor = randomTrainingExample()
      output = evaluate(line_tensor)
      guess, guess_i = categoryFromOutput(output)
      category_i = all_categories.index(category)
      confusion[category_i][guess_i] += 1

   # Normalize by dividing every row by its sum
   for i in range(n_categories):
      confusion[i] = confusion[i] / confusion[i].sum()

   # Set up plot
   fig = plt.figure()
   ax = fig.add_subplot(111)
   cax = ax.matshow(confusion.numpy())
   fig.colorbar(cax)

   # Set up axes
   ax.set_xticklabels([''] + all_categories, rotation=90)
   ax.set_yticklabels([''] + all_categories)

   # Force label at every tick
   ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
   ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

   # sphinx_gallery_thumbnail_number = 2
   plt.show()
   ```
----
#### Generating Names with A Character-Level RNN
----
   1. 构建网络, 在原有网络上进行修改![RNN](C:\Users\lenovo\Desktop\learn_pytorch\NLPwithPyTorch\RNN_forGeneratingName.png)  
   ```python
   import torch
   import torch.nn as nn

   class RNN(nn.Module):
      def __init__(self, input_size, hidden_size, output_size):
         super(RNN, self).__init__()
         self.hidden_size = hidden_size

         self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
         self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
         self.o2o = nn.Linear(hidden_size + output_size, output_size)
         self.dropout = nn.Dropout(0.1)
         self.softmax = nn.LogSoftmax(dim=1)

      def forward(self, category, input, hidden):
         input_combined = torch.cat((category, input, hidden), 1)
         hidden = self.i2h(input_combined)
         output = self.i2o(input_combined)
         output_combined = torch.cat((hidden, output), 1)
         output = self.o2o(output_combined)
         output = self.dropout(output)
         output = self.softmax(output)
         return output, hidden

      def initHidden(self):
         return torch.zeros(1, self.hidden_size)
   ```
   2. 训练
      + 准备数据
      ```python
      import random

      # Random item from a list
      def randomChoice(l):
         return l[random.randint(0, len(l) - 1)]

      # Get a random category and random line from that category
      def randomTrainingPair():
         category = randomChoice(all_categories)
         line = randomChoice(category_lines[category])
         return category, line
      ```
      + 该网络训练需要category, 需要input, 同时还需要一个target_output, 因为名字的产生是逐个字母, 所以需要每一个rnn cell的output, 同时还需要一个结束符号![RNNTraining](C:\Users\lenovo\Desktop\learn_pytorch\NLPwithPyTorch\training.png)
      ```python
      # One-hot vector for category
      def categoryTensor(category):
         li = all_categories.index(category)
         tensor = torch.zeros(1, n_categories)
         tensor[0][li] = 1
         return tensor

      # One-hot matrix of first to last letters (not including EOS) for input
      def inputTensor(line):
         tensor = torch.zeros(len(line), 1, n_letters)
         for li in range(len(line)):
            letter = line[li]
            tensor[li][0][all_letters.find(letter)] = 1
         return tensor

      # LongTensor of second letter to end (EOS) for target
      def targetTensor(line):
         letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
         letter_indexes.append(n_letters - 1) # EOS
         return torch.LongTensor(letter_indexes)

      # Make category, input, and target tensors from a random category, line pair
      def randomTrainingExample():
         category, line = randomTrainingPair()
         category_tensor = categoryTensor(category)
         input_line_tensor = inputTensor(line)
         target_line_tensor = targetTensor(line)
         return category_tensor, input_line_tensor, target_line_tensor
      ```
      + 训练网络
      ```python
      criterion = nn.NLLLoss()
      learning_rate = 0.0005

      def train(category_tensor, input_line_tensor, target_line_tensor):
         target_line_tensor.unsqueeze_(-1)
         hidden = rnn.initHidden()

         rnn.zero_grad()
         loss = 0
         for i in range(input_line_tensor.size(0)):
               output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
               l = criterion(output, target_line_tensor[i])
               loss += l
         loss.backward()

         for p in rnn.parameters():
            p.data.add_(learning_rate, p.grad.data)
         
         return output, loss.item() / input_line_tensor.size(0)
      ```
      + 训练
      ```python
      import time
      import math

      def timeSince(since):
         now = time.time()
         s = now - since
         m = math.floor(s / 60)
         s -= m * 60
         return '%dm %ds' % (m, s)
      
      rnn = RNN(n_letters, 128, n_letters)

      n_iters = 100000
      print_every = 5000
      plot_every = 500
      all_losses = []
      total_loss = 0 # Reset every plot_every iters

      start = time.time()

      for iter in range(1, n_iters + 1):
         output, loss = train(*randomTrainingExample())
         total_loss += loss

         if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

         if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0
      
      import matplotlib.pyplot as plt
      import matplotlib.ticker as ticker

      plt.figure()
      plt.plot(all_losses)
      ```
   3. Sampling the Network 测试网络
   ```python
   max_length = 20

   # Sample from a category and starting letter
   def sample(category, start_letter='A'):
      with torch.no_grad():  # no need to track history in sampling
         category_tensor = categoryTensor(category)
         input = inputTensor(start_letter)
         hidden = rnn.initHidden()

         output_name = start_letter

         for i in range(max_length):
               output, hidden = rnn(category_tensor, input[0], hidden)
               topv, topi = output.topk(1)
               topi = topi[0][0]
               if topi == n_letters - 1:
                  break
               else:
                  letter = all_letters[topi]
                  output_name += letter
               input = inputTensor(letter)

         return output_name

   # Get multiple samples from one category and multiple starting letters
   def samples(category, start_letters='ABC'):
      for start_letter in start_letters:
         print(sample(category, start_letter))

   samples('Russian', 'RUS')

   samples('German', 'GER')

   samples('Spanish', 'SPA')

   samples('Chinese', 'CHI')
   ```
----
#### Translation with Seq2Seq Network and Attention
----
   1. Load Datas 加载数据
      + word embedding, 使用one-hot编码, 建立一个转换的类
      ```python
      from __future__ import unicode_literals, print_function, division
      from io import open
      import unicodedata
      import string
      import re
      import random

      import torch
      import torch.nn as nn
      from torch import optim
      import torch.nn.functional as F

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      SOS_token = 0
      EOS_token = 1

      class Lang:
         def __init__(self, name):
            self.name = name
            self.word2index = {}
            self.word2count = {}
            self.index2word = {0: "SOS", 1: "EOS"}
            self.n_words = 2  # Count SOS and EOS
        
         def addSentence(self, sentence):
            for word in sentence.split(' '):
                  self.addWord(word)
        
         def addWord(self, word):
            if word not in self.word2index:
                  self.word2index[word] = self.n_words
                  self.word2count[word] = 1
                  self.index2word[self.n_words] = word
                  self.n_words += 1
            else:
                  self.word2count[word] += 1
      ```
      + 编码转换
      ```python
      # Turn a Unicode string to plain ASCII, thanks to
      # https://stackoverflow.com/a/518232/2809427
      def unicodeToAscii(s):
         return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
         )
        
      # Lowercase, trim, and remove non-letter characters
      
      def normalizeString(s):
         s = unicodeToAscii(s.lower().strip())
         s = re.sub(r"([.!?])", r" \1", s)
         s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
         return s
      ```
      + 读取数据
      ```python
      def readLangs(lang1, lang2, reverse=False):
         print("Reading lines...")
        
         # Read the file and split into lines
         lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
            read().strip().split('\n')
        
         # Split every line into pairs and normalize
         pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
        
         # Reverse pairs, make Lang instances
         if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
         else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)
        
         return input_lang, output_lang, pairs
      ```
      + 缩减句子
      ```python
      MAX_LENGTH = 10
        
      eng_prefixes = (
         "i am ", "i m ",
         "he is", "he s ",
         "she is", "she s ",
         "you are", "you re ",
         "we are", "we re ",
         "they are", "they re "
      )

      def filterPair(p):
         return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
            p[1].startswith(eng_prefixes)

      def filterPairs(pairs):
         return [pair for pair in pairs if filterPair(pair)]
      ```
   + The full process for preparing the data
      ```python
      def prepareData(lang1, lang2, reverse=False):
         input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
         print("Read %s sentence pairs" % len(pairs))
         pairs = filterPairs(pairs)
         print("Trimmed to %s sentence pairs" % len(pairs))
         print("Counting words...")
         for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
         print("Counted words:")
         print(input_lang.name, input_lang.n_words)
         print(output_lang.name, output_lang.n_words)
         return input_lang, output_lang, pairs
        
      input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
      print(random.choice(pairs))
      ```
   2. The Seq2seq Model, Encoder得到a single vector, 然后经过Decoder解码![Seq2seq](C:\Users\lenovo\Desktop\learn_pytorch\NLPwithPyTorch\seq2seq.png)
      + The Encoder![Encoder](C:\Users\lenovo\Desktop\learn_pytorch\NLPwithPyTorch\encoder-network.png) 
      ```python
      class EncoderRNN(nn.Module):
         def __init__(self, input_size, hidden_size):
            super(EncoderRNN, self).__init__()
            self.hidden_size = hidden_size
            self.emdedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)
         
         def initHidden(self):
            return torch.randn(1, 1, self.hidden_size, device=device)
         
         def forward(self, input, hidden):
            embedded = self.embedding(input).view(1, 1, -1)
            output = embedded
            output, hidden = self.gru(output, hidden)
            return output, hidden
      ```
      + The Decoder
         + The simple decoder![simpleDecoder](C:\Users\lenovo\Desktop\learn_pytorch\NLPwithPyTorch\decoder-network.png)
         ```python
         class DecoderRNN(nn.Module):
            def __init__(self, hidden_size, output_size):
               super(DecoderRNN, self).__init__()
               self.hidden_size = hidden_size
               self.emdedding = nn.Embedding(hidden_size, hidden_size)
               self.gru = nn.GRU(hidden_size, hidden_size)
               self.out = nn.Linear(hidden_size, output_size)
               self.softmax = nn.LogSoftmax(dim=1)
            
            def initHidden(self):
               return torch.randn(1, 1, self.hidden_size, device=device)
            
            def forward(self, input, hidden):
               output = self.emdedding(input).view(1, 1, -1)
               output = F.relu(output)
               output, hidden = self.gru(output, hidden)
               output = self.softmax(self.out(output[0]))
               return output, hidden
         ```
         + Attention Decoder
         + ![simpleDecoder](C:\Users\lenovo\Desktop\learn_pytorch\NLPwithPyTorch\attention-decoder-network.png)
         ```python
         class AttnDecoderRNN(nn.Module):
            def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
               super(AttnDecoderRNN, self).__init__()
               self.hidden_size = hidden_size
               self.output_size = output_size
               self.dropout_p = dropout_p
      self.max_length = max_length
         
               self.embedding = nn.Embedding(self.output_size, self.hidden_size)
               self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
               self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
               self.dropout = nn.Dropout(self.dropout_p)
               self.gru = nn.GRU(self.hidden_size, self.hidden_size)
      self.out = nn.Linear(self.hidden_size, self.output_size)
         
            def forward(self, input, hidden, encoder_outputs):
               embedded = self.embedding(input).view(1, 1, -1)
      embedded = self.dropout(embedded)
         
               attn_weights = F.softmax(
                     self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
               attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
         
               output = torch.cat((embedded[0], attn_applied[0]), 1)
      output = self.attn_combine(output).unsqueeze(0)
         
               output = F.relu(output)
      output, hidden = self.gru(output, hidden)
         
               output = F.log_softmax(self.out(output[0]), dim=1)
      return output, hidden, attn_weights
         
            def initHidden(self):
               return torch.zeros(1, 1, self.hidden_size, device=device)
         ```
   3. Training
      + Preparing the data
      ```python
      def indexesFromSentence(lang, sentence):
         return [lang.word2index[word] for word in sentence.split(' ')]
      
      def tensorFromSentence(lang, sentence):
         indexes = indexesFromSentence(lang, sentence)
         indexes.append(EOS_token)
         return torch.tensor(indexes, dtype=torch.long,   device=device).view(-1, 1)
      

      def tensorsFromPair(pair):
         input_tensor = tensorFromSentence(input_lang, pair[0])
         target_tensor = tensorFromSentence(output_lang, pair[1])
         return (input_tensor, target_tensor)
      ```
   + 为了训练我们通过编码器运行输入语句，并跟踪每个输出和最新的隐藏状态。然后，将译码器的`<SOS>`令牌作为其第一个输入，将编码器的最后一个隐藏状态作为其第一个隐藏状态。
   + 'Teacher Forcing'使用真实的目标输出作为下一个Cell输入，而不是使用解码器的此时的输出作为下一个输入。使用'Teacher Forcing'使其收敛得更快，但当训练好的网络被使用时，它可能会表现出不稳定性。
      ```python
      teacher_forcing_ratio = 0.5

      def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
         encoder_hidden = encoder.initHidden()
          
       encoder_optimizer.zero_grad()
         decoder_optimizer.zero_grad()
          
       input_length = input_tensor.size(0)
         target_length = target_tensor.size(0)
          
       encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
          
       loss = 0
          
       for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                  input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
          
       decoder_input = torch.tensor([[SOS_token]], device=device)
          
       decoder_hidden = encoder_hidden
          
       use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
          
       if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                  decoder_output, decoder_hidden, decoder_attention = decoder(
                     decoder_input, decoder_hidden, encoder_outputs)
                  loss += criterion(decoder_output, target_tensor[di])
                  decoder_input = target_tensor[di]  # Teacher forcing
          
       else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                  decoder_output, decoder_hidden, decoder_attention = decoder(
                     decoder_input, decoder_hidden, encoder_outputs)
                  topv, topi = decoder_output.topk(1)
                  decoder_input = topi.squeeze().detach()  # detach from history as input
          
                loss += criterion(decoder_output, target_tensor[di])
                  if decoder_input.item() == EOS_token:
                     break
          
       loss.backward()
          
       encoder_optimizer.step()
         decoder_optimizer.step()
          
       return loss.item() / target_length
      ```
      + 记录训练时间
      ```python
      import time
      import math

      def asMinutes(s):
         m = math.floor(s / 60)
         s -= m * 60
         return '%dm %ds' % (m, s)
          
      def timeSince(since, percent):
         now = time.time()
         s = now - since
         es = s / (percent)
         rs = es - s
         return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
      ```
      + 迭代训练
      ```python
      def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
         start = time.time()
         plot_losses = []
         print_loss_total = 0  # Reset every print_every
         plot_loss_total = 0  # Reset every plot_every
          
       encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
         decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
         training_pairs = [tensorsFromPair(random.choice(pairs))
                           for i in range(n_iters)]
         criterion = nn.NLLLoss()
          
       for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
          
          loss = train(input_tensor, target_tensor, encoder,
                           decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
          
          if iter % print_every == 0:
                  print_loss_avg = print_loss_total / print_every
                  print_loss_total = 0
                  print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))
          
          if iter % plot_every == 0:
                  plot_loss_avg = plot_loss_total / plot_every
                  plot_losses.append(plot_loss_avg)
                  plot_loss_total = 0
          
       showPlot(plot_losses)
      ```
      + 绘图结果
      ```python
      import matplotlib.pyplot as plt
      plt.switch_backend('agg')
      import matplotlib.ticker as ticker
      import numpy as np
      
      def showPlot(points):
         plt.figure()
         fig, ax = plt.subplots()
         # this locator puts ticks at regular intervals
         loc = ticker.MultipleLocator(base=0.2)
         ax.yaxis.set_major_locator(loc)
         plt.plot(points)
      ```
   4. Evaluation
      + 对某一个样本进行评估 
      ```python
      def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
      with torch.no_grad():
         input_tensor = tensorFromSentence(input_lang, sentence)
         input_length = input_tensor.size()[0]
         encoder_hidden = encoder.initHidden()

         encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

         for ei in range(input_length):
               encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                      encoder_hidden)
               encoder_outputs[ei] += encoder_output[0, 0]

         decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

         decoder_hidden = encoder_hidden

         decoded_words = []
         decoder_attentions = torch.zeros(max_length, max_length)

         for di in range(max_length):
               decoder_output, decoder_hidden, decoder_attention = decoder(
                  decoder_input, decoder_hidden, encoder_outputs)
               decoder_attentions[di] = decoder_attention.data
               topv, topi = decoder_output.data.topk(1)
               if topi.item() == EOS_token:
                  decoded_words.append('<EOS>')
                  break
               else:
                  decoded_words.append(output_lang.index2word[topi.item()])

               decoder_input = topi.squeeze().detach()

         return decoded_words, decoder_attentions[:di + 1]
      ```
      + 随机抽取样本进行评估
      ```python
      def evaluateRandomly(encoder, decoder, n=10):
         for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')
      ```
   5. 可视化Attention
   ```python
   def showAttention(input_sentence, output_words, attentions):
      # Set up figure with colorbar
      fig = plt.figure()
      ax = fig.add_subplot(111)
      cax = ax.matshow(attentions.numpy(), cmap='bone')
      fig.colorbar(cax)

      # Set up axes
      ax.set_xticklabels([''] + input_sentence.split(' ') +
                        ['<EOS>'], rotation=90)
      ax.set_yticklabels([''] + output_words)

      # Show label at every tick
      ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
      ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

      plt.show()


   def evaluateAndShowAttention(input_sentence):
      output_words, attentions = evaluate(
         encoder1, attn_decoder1, input_sentence)
      print('input =', input_sentence)
      print('output =', ' '.join(output_words))
      showAttention(input_sentence, output_words, attentions)


   evaluateAndShowAttention("elle a cinq ans de moins que moi .")

   evaluateAndShowAttention("elle est trop petit .")

   evaluateAndShowAttention("je ne crains pas de mourir .")

   evaluateAndShowAttention("c est un jeune directeur plein de talent .")
   ```


​    

   
