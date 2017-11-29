import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

temp = np.load('3__0__conv.npy.npz')

a = temp.items()
a=a[0]
a=a[1]
b = np.ones((64,32*32*128))
for i in range(128):
    for j in range(64):
        for k in range(32):
            for l in range(32):
                b[j,  ((i*32)+k)*32+l   ] = a[i,j,k,l]
np.random.shuffle(b)
print(b.shape)
data = b
data=data.transpose()

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from pycrayon import CrayonClient
import time

cc = CrayonClient(hostname="10.150.6.120")

try:
    cc.remove_experiment("AnalyzeConv3")
except:
    pass

try:
    OMIE = cc.create_experiment("AnalyzeConv3")
except:
    pass

##
## noise level one
## dimension 2
### z는 따로 추출
###
input_size = 64
hidden_size = 128
hidden_size_ = 64
num_classes = 1

num_epochs = 9
learning_rate = 0.000001
debug_mode = True


class Net(nn.Module):
    def __init__(self, input_size, hidden_size_, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_)
        self.relu = nn.ReLU()
        self.fc2_ = nn.Linear(hidden_size_, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2_(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = Net(input_size, hidden_size_, hidden_size, num_classes).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(300000):
    batch_size = 10
    for j in range(1000):
        optimizer.zero_grad()
        output_sigma = torch.Tensor(1).cuda()
        exp_output_sigma = torch.Tensor(1).cuda()
        output_sigma[0] = 0
        exp_output_sigma[0] = 0
        output_sigma = Variable(output_sigma)
        exp_output_sigma = Variable(exp_output_sigma)
        if j % 100 == 0:
            print(epoch, j)
        for i in range(batch_size):
          #  print("--")
            np.random.shuffle(data)
            x_random = np.copy(data[i,0:32])
            z_random =  np.copy(data[i,32:64])

            x_random_margin = np.copy(x_random)
            z_random_margin = np.copy(data[batch_size+i,32:64])

            inputs = Variable(torch.from_numpy(np.concatenate((x_random, z_random))).cuda()).type(
                torch.cuda.FloatTensor)
            inputs2 = Variable(torch.from_numpy(np.concatenate((x_random_margin, z_random_margin))).cuda()).type(
                torch.cuda.FloatTensor)

            output_sigma = output_sigma + model(inputs)
            exp_output_sigma = exp_output_sigma + torch.exp(model(inputs2))

        loss = (output_sigma / batch_size) - torch.log(exp_output_sigma / batch_size)
        loss = -10 * loss
        OMIE.add_scalar_value("OMIE", np.float(loss.cpu().data.numpy()[0]))
        # loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
