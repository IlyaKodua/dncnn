from torch import nn
import torch.nn.functional as F
import torch

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 3)
        self.conv2 = nn.Conv1d(128,32,3)
        self.convT1 = nn.ConvTranspose1d(32,128 ,3)
        self.convT2 = nn.ConvTranspose1d(128,1 ,3)
        self.conv3 = nn.Conv1d(1,1,1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.convT1(x))
        x = F.relu(self.convT2(x))
        x = F.sigmoid(self.conv3(x))
        return x





