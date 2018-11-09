import torch

from torch import nn
from torch.nn import functional as F, Conv2d, Linear


class MultiResConv(nn.Module):
    def __init__(self, in_f, out_f, s=1):
        super(MultiResConv, self).__init__()

        windows = (1, 2, 3)
        assert out_f % len(windows) == 0, "out_f must be multiple of {}, got {} instead".format(len(windows), out_f)
        out_f //= len(windows)

        self.conv = nn.ModuleList([
            nn.Conv1d(in_f, out_f, kernel_size=(2 * w + 1), padding=w, stride=s)
            for w in windows
        ])

    def forward(self, x):
        return torch.cat([c(x) for c in self.conv], dim=1)


class RectCNN(nn.Module):
    def __init__(self, in_f=513):
        super(RectCNN, self).__init__()

        self.conv1 = MultiResConv(in_f, 96)
        self.conv2 = MultiResConv(96, 96, s=2)
        self.conv3 = MultiResConv(96, 96, s=2)
        self.conv4 = MultiResConv(96, 96, s=2)
        self.conv5 = nn.Conv1d(96, 7, kernel_size=1, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = x.mean(2)
        return x
        # x = F.relu(x)


class PaperCNN(nn.Module):
    def __init__(self):
        super(PaperCNN, self).__init__()
        self.conv1 = Conv2d(1, 16, kernel_size=3)
        self.conv2 = Conv2d(16, 32, kernel_size=3)
        self.conv3 = Conv2d(32, 64, kernel_size=3)
        self.conv4 = Conv2d(64, 64, kernel_size=3)
        self.conv5 = Conv2d(64, 128, kernel_size=3)
        self.conv6 = Conv2d(128, 128, kernel_size=3)
        # self.fc1 = Linear(128*14*5, 512)  # non-downsampled input size
        self.fc1 = Linear(128*6*2, 512)  # 0.55 scaled input
        self.fc2 = Linear(512, 7)

    def forward(self, x):
        x = x.unsqueeze(1)  # add channels dimension
        # BLOCK 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.25)

        # BLOCK 2
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.25)

        # BLOCK 3
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.25)

        # BLOCK 4
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = F.dropout(x, 0.25)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    import numpy as np
    from torchvision.datasets import DatasetFolder

    load = lambda x: torch.from_numpy(np.load(x))
    dataset = DatasetFolder('data/train', load, ('.npy',))

    x = dataset[0][0].unsqueeze(0).repeat(3,1,1)
    model = PaperCNN().double()
    y = model(x)
    print(y.shape)