import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
mean = np.array([1,2,3,1])
cov =  np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
# = np.random.multivariate_normal(mean,cov,5000)

from pycrayon import CrayonClient
import time

cc = CrayonClient(hostname="10.150.6.120")
cc.remove_experiment("OMIE_5")
OMIE = cc.create_experiment("OMIE_5")
###
### noise level one
### dimension 2
### z는 따로 추출
###
input_size = 4
hidden_size = 8
hidden_size_ = 3
num_classes = 1

num_epochs = 9
learning_rate = 0.0001
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
    batch_size = 40
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
            data_= np.random.multivariate_normal(mean, cov, 1)
            x_random = data_[0,0:2]
            z_random =  data_[0,2:4]

            data_= np.random.multivariate_normal(mean, cov, 1)
            x_random_margin = x_random
            z_random_margin = data_[0,2:4]

            inputs = Variable(torch.from_numpy(np.concatenate((x_random, z_random))).cuda()).type(
                torch.cuda.FloatTensor)
            inputs2 = Variable(torch.from_numpy(np.concatenate((x_random_margin, z_random_margin))).cuda()).type(
                torch.cuda.FloatTensor)

            output_sigma = output_sigma + model(inputs)
            exp_output_sigma = exp_output_sigma + torch.exp(model(inputs2))

        loss = (output_sigma / batch_size) - torch.log(exp_output_sigma / batch_size)
        loss = -10 * loss
        OMIE.add_scalar_value("accuracy", np.float(loss.cpu().data.numpy()[0]))
        # loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
