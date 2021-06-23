
import matplotlib.pyplot as plt
import helper

import Model
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import glob
import  random


dir_path = 'dataset/'

transform = transforms.Compose ([transforms.Resize (255), 
                                transforms.CenterCrop (224), 
                                transforms.ToTensor ()])


dataset = datasets.ImageFolder(root=dir_path, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

model = Model.AutoEncoder(3, 64, 3)
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = len(dataloader)
loss_arr = np.zeros((epochs,1))
cnt = 0
for images in dataloader:
    im = images[0]
    optimizer.zero_grad()
    noise = torch.zeros(im.size())
    stdN = np.random.uniform(0, 55, size=noise.size()[0])
    for n in range(noise.size()[0]):
        sizeN = noise[0,:,:,:].size()
        noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
    im_in = im + noise
    out = model(im_in)
    loss = criterion(out, noise) / (im.size()[0]*2)
#    for i in range(im_in.size()[0]):
#        im_out = im_in[i] - out[i]
#        cv2.imwrite('data_d/'+str(cnt) + "_" + str(i) + ".png" ,(im_out-im_out.min() ) / im_out.max())
#        cv2.imwrite('data_n/'+str(cnt) + "_" + str(i) + ".png"  ,(im_in[i]-im_in[i].min()) / im_in[i].max())
    loss_arr[cnt] = loss.cpu().detach().numpy().sum()
    loss.backward()
    optimizer.step()
    print(cnt)
    print(loss_arr[cnt])
    cnt += 1
PATH = 'net.pth'
torch.save(model.state_dict(), PATH)

plt.figure(2)
plt.plot(loss_arr, 'r')
plt.show()
cv2.waitKey(0)

