import torch
A_MODE_RESUME = False
A_Learning_Rate = 0.1
use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch