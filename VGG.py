import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 10),
        )

    def forward(self, X):
        h = self.conv1_1(X)
        h = F.relu(h)
        h = self.bn1(h)
        # -----------------------------------------
        h = self.conv1_2(h)
        h = F.relu(h)
        h = self.bn1(h)
        h = F.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))

        h = self.conv2_1(h)
        h = F.relu(h)
        h = self.bn2(h)
        # -----------------------------------------
        h = self.conv2_2(h)
        h = F.relu(h)
        h = self.bn2(h)
        h = F.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))

        h = self.conv3_1(h)
        h = F.relu(h)
        h = self.bn3(h)
        # -----------------------------------------
        h = self.conv3_2(h)
        h = F.relu(h)
        h = self.bn3(h)
        h = F.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        h_ = h
        #
        # h = self.conv4_1(h)
        # h = F.relu(h)
        # h = self.bn4(h)
        # # -----------------------------------------
        # h = self.conv4_2(h)
        # h = F.relu(h)
        # h = self.bn4(h)
        # h = F.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        #
        # h = self.conv5_1(h)
        # h = F.relu(h)
        # h = self.bn4(h)
        # # -----------------------------------------
        # h = self.conv5_2(h)
        # h = F.relu(h)
        # h = self.bn4(h)
        # h = F.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        # # ---------------------------------------------------------------
        # # ---------------------------------------------------------------
        # # ---------------------------------------------------------------
        #
        # h = h.view(h.size(0), -1)
        # h = self.classifier(h)
        h_ = h_.view(h_.size(0),-1)
        h_=self.classifier2(h_)
        return h_
