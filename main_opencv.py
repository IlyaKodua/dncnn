import cv2

import Model
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import glob
import random
import  math




dir_list = glob.glob('/home/liya/devel/DNCNN/AutorncoderSignal/dataset/train/*.png')

tran = transforms.ToTensor()

net = Model.AutoEncoder(1, 64, 3)
net = net.float()
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

epochs = len(dir_list)
loss_arr = np.zeros((epochs,1))
psnr_arr = np.zeros((epochs,1))
cnt = 0
for i in range(epochs):
    optimizer.zero_grad()
    im = cv2.imread(dir_list[i], 0)
    row, col = im.shape
    mean = 0
    var = random.uniform(0,55)
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = im + gauss.astype(np.uint8)
    noisy = torch.reshape(tran(noisy), ( 1, 1, col, row))
    gauss = torch.reshape(tran(gauss), ( 1, 1, col, row))
    im2 = torch.reshape(tran(im), ( 1, 1, col, row))
    out = net(noisy.float())
    loss = criterion(out, im2.float())
    loss_arr[cnt] = loss.item()
    psnr_arr[cnt] = 20 * math.log10(255/math.sqrt(loss_arr[cnt]) )
    loss.backward()
    optimizer.step()
    print(cnt)
    print(loss_arr[cnt])
    print(psnr_arr[cnt])
    cnt += 1
PATH = 'net2.pth'
torch.save(net.state_dict(), PATH)

plt.figure(2)
plt.plot(loss_arr, 'r')
plt.show()

plt.figure(3)
plt.plot(psnr_arr, 'b')
plt.show()