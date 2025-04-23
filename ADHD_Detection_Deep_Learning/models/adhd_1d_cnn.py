import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_ADHD(nn.Module):
    def __init__(self):
        super(CNN_ADHD, self).__init__()

        # Layer 1: Conv1D
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=27, stride=1)
        self.bn1 = nn.BatchNorm1d(3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2: Conv1D
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=10, kernel_size=15, stride=1)
        self.bn2 = nn.BatchNorm1d(10)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 3: Conv1D
        self.conv3 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm1d(10)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 4: Conv1D
        self.conv4 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm1d(10)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.4)

        # Fully connected layers
        self.fc1 = nn.Linear(1820, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)

        return x
