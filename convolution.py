'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from torch.autograd import Variable
from VGG import Net
from GV import *
import numpy as np
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
  #  transforms.RandomCrop(128, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if A_MODE_RESUME:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else :
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    net = Net()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=A_Learning_Rate, momentum=0.9, weight_decay=5e-4)



def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
       # print(inputs.data.shape)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print(loss)
      #  progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
       #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    #    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
     #       % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)


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
