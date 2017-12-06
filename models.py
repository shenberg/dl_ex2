
import torch
import torch.nn as nn


class Net12(nn.Module):
    def __init__(self):
        super().__init__()

        # assuming input size of 3x12x12
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(p=0.1, inplace=True)
        # output size 16x12x12
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # output size 16x6x6
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(16*6*6, 16)
        self.output = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        h = self.conv(x)
        h = self.dropout(h)
        h = self.pool(h)
        h = self.relu(h)
        # regular fc
        h = self.fc(h.view(-1, 16*6*6))
        h = self.relu(h)
        return self.sigmoid(self.output(h))


class Net12FCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # FCs converted to convolutions
        self.fc = nn.Conv2d(16, 16, kernel_size=6, stride=1)
        self.output = nn.Conv2d(16,1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h)
        h = self.relu(h)
        # fc as convolution - more natural too
        h = self.fc(h)
        h = self.relu(h)
        return self.sigmoid(self.output(h))

    def load_from_net12(self, path):
        "load state from saved Net12 - adapt weights to conv format"
        state_dict = torch.load(path)['state_dict']
        state_dict['fc.weight'] = state_dict['fc.weight'].view(16,16,6,6)
        state_dict['output.weight'] = state_dict['output.weight'].view(1,16,1,1)
        self.load_state_dict(state_dict)
        
class Net24(nn.Module):
    def __init__(self):
        super().__init__()

        # assuming input size of 3x24x24
        self.conv = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)

        self.dropout1 = nn.Dropout2d(p=0.5, inplace=True)
        # output size 64x24x24
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # output size 64x12x12
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=0.5, inplace=False)
        self.fc = nn.Linear(64*12*12, 128)
        self.output = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        h = self.conv(x)
        h = self.dropout1(h)
        h = self.pool(h)
        h = self.relu(h)
        h = self.dropout2(h)
        # regular fc
        h = self.fc(h.view(-1, 64*12*12))
        h = self.relu(h)
        return self.sigmoid(self.output(h))
