from torch import nn
import torch.nn.functional as F
import torch
import torchvision
import numpy as np

class AutoEncoder(nn.Module):

    def __init__(self, n_channels, n_filters, kernel_size):

        super(AutoEncoder, self).__init__()


        layers = [
            nn.Conv2d(in_channels=n_channels, out_channels=n_filters, kernel_size=kernel_size,
                      padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]

        depth = 20
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size,
                                    padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_filters))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_filters, out_channels=n_channels, kernel_size=kernel_size,
                                padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)



    def forward(self,x):
        out = self.dncnn(x)
        return out



